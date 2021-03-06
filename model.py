'''
Dataset,score,      time
A,      0.8821,     26.31732726097107    
B,      0.7536,     9.116921663284302
C,      0.9454,     6752.92465209960  (note: default max runtime is 200)
D,      0.9276,     202.9165310859680 (note: default max runtime is 200)
E,      0.8809,     196.8628749847412 (note: default max runtime is 100)
'''

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
import gc 

# Generic NN layers
from torch.nn import Linear, Bilinear

# Graph Learning layers to build our own models:
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TAGConv

# Graph Learning Models:
from torch_geometric.nn import Node2Vec

# Dimensionality reduction 
from sklearn.feature_selection import VarianceThreshold

# Balancing samples
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from math import log, e

from torch_geometric.data import Data
from torch_geometric.utils import degree

import copy

import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

class GAT(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=8, features_num=16, num_class=2, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        #self.lin1 = Linear(features_num, features_num//2)
        #self.lin2 = Linear(features_num//2, features_num//4)
        
        self.dropout_p = dropout 
        self.conv1 = GATConv(features_num, hidden, heads=heads, dropout=0.6, concat=False)
        
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden, num_class, heads=1, dropout=0.6)

    def forward(self, data):
        #x = F.dropout(data.x, p=0.6, training=self.training)
        #x = F.relu(self.lin1(data.x))
        #x = F.dropout(x, p=0.6, training=self.training)
        
        #x = F.relu(self.lin2(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__
    
# GCN model
class GCN(torch.nn.Module):
    
    def __init__(self, num_layers=2, hidden=16,  features_num=16, num_class=2):
        super(GCN, self).__init__()
        # first layer
        self.conv1 = GCNConv(features_num, hidden)
  
        # list of 2nd - num_layers layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        # fully connected layers
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        # clear weights
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # fully connected layer + relu
        x = F.relu(self.first_lin(x))
 
        # dropout layer
        x = F.dropout(x, p=0.5, training=self.training)

        # GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))

        # Another dropout
        x = F.dropout(x, p=0.5, training=self.training)

        # second FC layer
        x = self.lin2(x)

        # Softmax
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def generate_pyg_data(self, data): 
        has_feats = True
        ADD_GRAPH_FEATS = True

        # Load Feature Table
        x = data['fea_table']
        print("Num Nodes: %d" % x.shape[0])
        print("Node Features: %d" % (int(x.shape[1])-1))

        if x.shape[1] == 1: 
            print("No Features... using 1-hot encoding of node ID")
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
            has_feats = False
            
        else:
            # Otherwise we have features, so we drop the node_index comlumn
            x = x.drop('node_index', axis=1).to_numpy()
            
            print("Max Feature:")
            print(x.max())
            print("Min Feature:")
            print(x.min())
            
            # weird case with 0 features
            if x.min() == x.max():
                # all features are 0
                # so these certainly are not useful...
                # revert to node ID 1-hot vector
                print("Features are all zero!  Reverting to 1-hot node ID feature vectors...")
                x = data['fea_table']
                x = x.to_numpy(dtype=np.int32)
                node_ids = x[:,0]
                x = np.array(pd.get_dummies(node_ids))
                has_feats = False
        
        # Load edge data
        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        
        # Convert input data to tensor
        x = torch.tensor(x, dtype=torch.float)
        
        # Load the graph data
        print('Sorting edges')
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        print("Num edges: %d" % edge_index.size()[1])
        
        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # Filter out zero-weight edges
        edge_index = edge_index[:, edge_weight != 0]
        edge_weight = edge_weight[edge_weight != 0]
        print("Num non-zero edges: %d" % edge_index.size()[1])
        
        # Use torch geometric function; it's faster than building an nx graph
        if ADD_GRAPH_FEATS:
            print("Adding graph feats")
            graph_feats = degree(edge_index[0], num_nodes=x.size()[0])
            graph_feats = graph_feats / graph_feats.max()
            graph_feats = graph_feats.reshape(graph_feats.size()[0], 1)
        
        # This is a very computationally expensive line of code.
        # print("Max/Min edge weight: %f/%f" % (max(edge_weight), min(edge_weight)))
        
        # Build training labels
        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        # Quick stats on label data
        num_classes=int(max(y)) + 1 
        print("Number of classes: %d" % num_classes)
        per_class = [ 0 ] * num_classes
        for label in y[inds]:
            per_class[label] += 1
        
        print("Samples per class:")
        print(per_class)
        per_class_percent = [ (a*100)/len(inds) for a in per_class ]
        print("Class Distribution:")
        print(per_class_percent)
        
        # Use beta-normalized weights
        BETA = 1-(10 ** -2)
        per_class = np.array(per_class)
        effective_num = 1.0 - np.power(BETA, per_class)
        weights = (1.0 - BETA) / effective_num
        weight_vector = weights / np.sum(weights) * len(per_class)
        
        print("Weight vector for weighted loss calcs:")
        print(weight_vector)
        
        # Build train,validate, and test masks
        all_train_indices = data['train_indices']
        print("Num all training: %d" % len(all_train_indices))
        
        '''
        # Remove singletons
        all_train = torch.zeros(num_nodes, dtype=torch.bool)
        all_train[all_train_indices] = 1
        all_train[graph_feats.reshape(graph_feats.size()[0]) == 0] = 0
        all_train_indices = all_train.nonzero().numpy()
        '''
        
        # More intelligent split that keeps all classes in both splits,
        # and keeps the same distr of classes
        train_indices, val_indices = train_test_split(
            all_train_indices, 
            test_size=0.25,
            stratify=y[all_train_indices].numpy(),
            shuffle=True
        )
        
        # train_indices = self.resample(train_indices, y, np.argmax(per_class))
        
        print("Num train nodes: %d" % len(train_indices))
        print("Num val nodes: %d" % len(val_indices))  
        
        # Lets check our validation set and make sure it looks reasonable:
        print("Validation Set Distribution:")
        per_class = [ 0 ] * num_classes
        for label in y[val_indices]:
            per_class[label] += 1
        print("Samples per class:")
        print(per_class)
        per_class_percent = [ (a*100)/len(val_indices) for a in per_class ]
        print("Class Distribution:")
        print(per_class_percent)

        test_indices = data['test_indices']
        print("Num test nodes: %d" % len(test_indices))

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        if ADD_GRAPH_FEATS:
            data.graph_data = graph_feats
            
        data.num_nodes = num_nodes
        data.per_class_percent = per_class_percent

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_indices] = 1
        data.val_mask = val_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask

        data.has_features = has_feats
        data.weighted_loss = torch.tensor(weight_vector, dtype=torch.float)

        return data

    def train(self, data, start_time, time_budget):        
        ADD_N2V = False
        ADD_GRAPH_FEATS = True
        MIN_TIME = 3 # Stop training loop early if less than this many seconds remain
        
        # Graph data
        avg_degree = degree(data.edge_index[0], data.x.size()[0]).mean()
        
        # Add n2v embeddings to features if there are an order of magnitude more
        # edges than there are features
        if int(log(data.x.size()[0], 10)) < int(log(data.edge_index.size()[1], 10)):
            ADD_N2V = True
        
        # Hyperparamters
        train_epochs=1000
        num_layers=2 # gcn layers
        
        # Different algorithms for the number of hidden dims for each 
        if data.has_features:
            hidden = min([int(max(data.y)+1) ** 2, 128])
            attn_heads = 'N/a'
        else:
            attn_heads = min([int(log(max(data.y)+1)) + 2, 4])
            hidden = (min([int(max(data.y)+1) ** 2, 32]) // attn_heads) + 1
            
        early_stopping=True
        val_patience = 100 # how long validation loss can increase before we stop
        
        # Use heuristic-based hyperparams if too many edges to handle
        simplified = True if data.edge_index.size()[1] > 1e6 else False
        
        print('Hidden dimensions: %d' % hidden)
        print('Attention heads: %s' % str(attn_heads))
        if not data.has_features or ADD_N2V:
            # Requires at least len(class) dimensions, but give it a little more
            embedding_dim = 128 + int(avg_degree ** (1/2))
            
            # The larger the avg degree, the less distant walks matter
            # Of course, a minimum is still important
            context_size = int(log(data.edge_index.size()[1])/avg_degree)
            context_size = context_size if context_size >= 3 else 3
            
            # We should look at at least 1 context per walk
            walk_len = context_size + 1
        
            print('Embedding dim: %d\tWalk Len: %d\tContext size: %d'
                  % (embedding_dim, walk_len, context_size))
        
            embedder = Node2Vec(
                data.x.size()[0],   # Num nodes
                embedding_dim,      # Embedding dimesion
                walk_len,           # Walk len  
                context_size,        # Context size 
                num_negative_samples=context_size**2
            )
            
            # First, train embedder
            # Use a higher learning rate, bc this part is
            # meant to be kind of "quick and dirty"
            embedder = self.n2v_trainer(
                data, embedder, lr=0.05, patience=50 # lower patience when time is important
            )
            
            # Training moves data to GPU. Have to put it back before manipulating
            # it further. 
            data = data.to('cpu')
            embedder = embedder.to('cpu')
            
            if data.has_features and ADD_N2V:
                data.x = torch.cat((self.var_thresh(data.x), embedder.embedding.weight), axis=1)
            else:
                # Then use n2v embeddings as features
                data.x = embedder.embedding.weight
            
            # Remove reference to embedder to free up memory on GPU
            del embedder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
        
        else:
            print('Num feature before: %d' % data.x.size()[1])
            data.x = self.var_thresh(data.x)
            print('Num features after: %d' % data.x.size()[1])

        if ADD_GRAPH_FEATS:
            print('Num feature before: %d' % data.x.size()[1])
            data.x = torch.cat((data.x, data.graph_data), axis=1)
            print('Num features after: %d' % data.x.size()[1])
             
        if data.has_features:
            print("Using GCN")
            
            # Make sure we actually need this.. Only if it crashes on test data
            # Just use heuristics-based 
            #if simplified:
            #    params = {
            #        'features_num': data.x.size()[1],
            #        'num_class': int(max(data.y)) + 1,
            #        'hidden': hidden
            #    }
            
            # Grid search to find best
            #else:
            params = self.grid_search(
                data,
                hidden,
                h_dist=10,
                epochs=50,
                h_step=1
            )
            model = GCN(**params)
          
        else:
            print("Using GAT")
            
            # Just use heuristics based 
            if simplified:
                params = {
                    'features_num': data.x.size()[1],
                    'num_class': int(max(data.y)) + 1,
                    'hidden': hidden,
                    'heads': attn_heads,
                    'dropout': 0.7 # Increase dropout if complex graph
                }
                
            # Do grid search for best params
            else:
                params = self.grid_search(
                    data,
                    hidden,
                    attn_heads=attn_heads,
                    h_dist=10,
                    h_step=1,
                    a_dist=1,
                    a_step=1,
                    epochs=25
                )
                
            model = GAT(**params)

        # Move data to compute device
        model = model.to(self.device)
        data = data.to(self.device)

        # Configure optimizer
        lr = 0.005
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
  
        # Main training loop
        min_loss = float('inf')
        val_loss_min = 1000
        train_loss_min = 1000
        val_increase=0
        stopped_early=False
        state_dict_save = 'checkpoint.model'
        
        # Fraction of patience before restarting with lower lr from best model
        FRUSTRATION = 1 if data.has_features else 25
        # Number of times to retry training from prev best with lower lr
        NUM_REDOS = 15
        # Smallest value we allow loss to be, usually 5e-6
        MIN_LOSS = 5e-6
        # How much we decrease loss when frustrated
        LOSS_DEC_INC = 1.25
        
        # What percent validation loss increase we're willing to accept if training loss
        # goes down by more than that much. The hope is this balances for overfitting?
        # E.g., an epoch that increases val loss from 1 to 1.01 but decreases train loss
        # from 1 to 0.98 is considered the best model 
        GOOD_ENOUGH = 1.00025
        BECOME_FRUSTRATED = False
        lr_decays = 0
        
        # LR must decay at least this many times before GOOD_ENOUGH training
        # is activated. This way we're sure it's in a local minimum before we
        # allow it to stray 
        GOOD_ENOUGH_THRESH = 3
        
        epoch = 0
        while(True):
            train_start = time.time()
            
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(
                model(data)[data.train_mask], 
                data.y[data.train_mask], 
                weight=data.weighted_loss
            )
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            
            # calculate loss on validation set
            model.eval()
            loss = F.nll_loss(
                model(data)[data.val_mask], 
                data.y[data.val_mask],
                weight=data.weighted_loss
            )
            
            val_loss = loss.item()
            
            if ((val_loss > val_loss_min and early_stopping) and 
                not (BECOME_FRUSTRATED and val_loss <= GOOD_ENOUGH*val_loss_min and train_loss*GOOD_ENOUGH <= train_loss_min)):
                val_increase+= 1
            else:
                print("===New Minimum validation loss===")
                print('[%d] Train loss: %.3f   Val Loss: %.3f' % (epoch, train_loss, val_loss))
                val_loss_min = val_loss
                train_loss_min = train_loss
                val_increase=0
                redos=1
                torch.save(model.state_dict(), state_dict_save)
            
            # Want to make sure we have the amount of time it takes
            # to loop and however much extra we need later
            time_cutoff = MIN_TIME + (time.time() - train_start)
            
            if (val_increase > val_patience or 
                time_budget - (time.time() - start_time) < time_cutoff):
                
                print("Early stopping!")
                stopped_early=True
                break
            
            
            # Lower learning rate and start from prev best after model becomes 
            # frustrated with poor progress
            if val_increase > val_patience//FRUSTRATION and lr > MIN_LOSS:
                if redos % NUM_REDOS == 0:
                    lr /= LOSS_DEC_INC
                    lr = lr if lr > MIN_LOSS else MIN_LOSS # make sure not less than 5e-6
                    print('LR decay: New lr: %.6f' % lr)
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    
                    lr_decays += 1
                    if lr_decays >= GOOD_ENOUGH_THRESH:
                        BECOME_FRUSTRATED = True
                        
                    model.load_state_dict(torch.load(state_dict_save))
                
                redos += 1
                
            # Use simple LR decay for data w features
            elif data.has_features and epoch > 0 and epoch % 10 == 0 and lr > 5e-6:
                lr -= 0.00025
                lr = lr if lr > 5e-6 else 5e-6 # make sure not less than 5e-6
                print('LR decay: New lr: %.6f' % lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr
            
            epoch += 1
            
        if stopped_early:
            print("Reloading best parameters!")
            
            # State dict saved to CPU so have to load from there(?)
            model.load_state_dict(torch.load(state_dict_save))
            
        return model

    def pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred

    def train_predict(self, data, time_budget,n_class,schema):
        start_time = time.time()
        
        data = self.generate_pyg_data(data)
        model = self.train(data, start_time, time_budget)
        pred = self.pred(model, data)

        return pred.cpu().numpy().flatten()

    def grid_search(self, data, hidden, attn_heads=None, h_dist=10, 
                    h_step=2, a_dist=0, a_step=0, epochs=15):
        
        # Best parameters for the model
        best = {
            'features_num': data.x.size()[1],
            'num_class': int(max(data.y)) + 1
        }
        saved = 'best_params.model'
        
        if attn_heads:
            attn_range = range(max(2, attn_heads-a_dist), attn_heads+a_dist+1, a_step)
        else:
            attn_range = range(1)
        
        # Want at least 4
        hidden_range = range(max(4,hidden-h_dist), hidden+h_dist, h_step)
        print('Hidden range: %s\nAttn Range: %s' % (str(list(hidden_range)), str(list(attn_range)) ))
        
        min_loss = float('inf')
        data.to(self.device)
        for h in hidden_range:
            oom = False 
            for a in attn_range:
                lr = 0.005
            
                if attn_heads:
                    model = GAT(
                        features_num=data.x.size()[1],
                        num_class=int(max(data.y)) + 1, 
                        hidden=h,
                        heads=a
                    )
                else:
                    model = GCN(
                        features_num=data.x.size()[1],
                        num_class=int(max(data.y)) + 1, 
                        hidden=h
                    )
                
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=1e-5
                )
                
                model.to(self.device)
                print("Testing %d hidden dims and %d attn heads" % (h,a))
                
                # Do short training
                min_val_loss = float('inf')
                for epoch in range(epochs):
                    try: 
                        model.train()
                        optimizer.zero_grad()
                        loss = F.nll_loss(
                            model(data)[data.train_mask], 
                            data.y[data.train_mask], 
                            weight=data.weighted_loss
                        )
                        loss.backward()
                        optimizer.step()
                        
                        # LR decay for GCN
                        if not attn_heads and epoch > 0 and epoch % 10 == 0 and lr > 5e-6:
                            lr -= 0.00025
                            lr = lr if lr > 5e-6 else 5e-6 # make sure not less than 5e-6
                            
                            for g in optimizer.param_groups:
                                g['lr'] = lr
                                
                        model.eval()
                        loss = F.nll_loss(
                            model(data)[data.val_mask], 
                            data.y[data.val_mask],
                            weight=data.weighted_loss
                        )
                        
                        val_loss = loss.item()
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                                
                                
                    except RuntimeError as e:
                        print(e)
                        del model 
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        oom = True
                    
                    if oom:
                        break
                  
                # If we rean out of memory, don't check any
                # larger parameters      
                if oom:
                    break 
                
                # calculate loss on validation set
                print("%d hidden\t%d attn heads\t%0.3f loss" % (h, a, min_val_loss))
                if min_val_loss < min_loss:
                    print("===New best params===")
                    min_loss = min_val_loss
                    best['hidden'] = h
                    
                    if attn_heads:
                        best['heads'] = a
                    
                    
                # Free as much memory as possible between models
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        return best

    
    def n2v_trainer(self, data, model, epochs=800, early_stopping=True, 
                    patience=10, verbosity=1, lr=0.01):
        
        print("Training n2v")
        model = model.to(self.device)
        data = data.to(self.device)
        
        stopped_early = True
        loss_min=1000
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        increase = 0
        epoch = -1
        
        redos = 0
        
        # Fraction of patience before restarting with lower lr from best model
        FRUSTRATION = 1 if data.has_features else 10
        # Number of times to retry training from prev best with lower lr
        NUM_REDOS = 25
        # Smallest value we allow loss to be, usually 5e-6
        MIN_LOSS = 5e-6
        # How much we decrease loss when frustrated
        LOSS_DEC_INC = 1.25
        
        while(True):
            model.train()
            
            optimizer.zero_grad()
            loss = model.loss(data.edge_index)
            loss.backward()
            optimizer.step()
            epoch += 1
            
            if loss > loss_min and early_stopping:
                increase+= 1
            else:
                if verbosity > 0:
                    print("===New Minimum loss===")
                    print('[%d] Loss: %.3f' % (epoch, loss))
                loss_min = loss
                increase=0
                redos=0
                state_dict_save = copy.deepcopy(model.state_dict())
            if increase > patience:
                print("Early stopping!")
                stopped_early=True
                break
            
            # Lower learning rate and start from prev best after model becomes 
            # frustrated with poor progress
            if increase > patience//FRUSTRATION and lr > MIN_LOSS:
                if redos % NUM_REDOS == 0:
                    lr /= LOSS_DEC_INC
                    lr = lr if lr > MIN_LOSS else MIN_LOSS # make sure not less than 5e-6
                    print('LR decay: New lr: %.6f' % lr)
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                        
                    model.load_state_dict(state_dict_save)
                    redos = 0
                
                redos += 1
                
            # For data w features, use simple LR decay
            elif data.has_features and epoch > 0 and epoch % 10 == 0 and lr > 5e-6:
                lr -= 0.00025
                lr = lr if lr > 5e-6 else 5e-6 # make sure not less than 5e-6
                print('LR decay: New lr: %.6f' % lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr
            
                
        if stopped_early:
            print("Reloading best parameters!")
            model.load_state_dict(state_dict_save)
            
        return model
    
    def var_thresh(self, x, var=0.0):
        sel = VarianceThreshold(var)
        
        x = torch.tensor(sel.fit_transform(x), dtype=torch.float)
        return x

