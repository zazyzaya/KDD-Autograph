'''
Dataset,score,      time
A,      0.8840,     24.511635780334473    
B,      0.7481,     7.5256383419036865
C,      0.8642,     85.26456379890442
D,      0.9276,     202.91653108596802 (note: default max runtime is 200)
E,      0.8818,     143.4068477153778 (note: default max runtime is 100)
'''

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Generic NN layers
from torch.nn import Linear

# Graph Learning layers to build our own models:
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

# Graph Learning Models:
from torch_geometric.nn import JumpingKnowledge, Node2Vec

# Minibatching
from torch_geometric.data import NeighborSampler

# Dimensionality reduction 
from sklearn.feature_selection import VarianceThreshold

# Balancing samples
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from math import log
import networkx as nx

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


class JustFeatures(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2):
        super().__init__()
        
        self.first_lin = Linear(features_num, hidden)
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(Linear(hidden, hidden))
        
        self.last_lin = Linear(hidden, num_class)
    
    def reset_parameters(self):
        self.first_lin.reset_parameters()
        for l in self.hidden_layers:
            l.reset_parameters()
        self.last_lin.reset_parameters()
        
    def forward(self, data):
        x = data.x 
        
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for l in self.hidden_layers:
            x = F.relu(l(x))
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_lin(x)
        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=8, features_num=16, num_class=2, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(features_num, hidden, heads=heads, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden * heads, num_class, heads=1, concat=True,dropout=0.6)

    def forward(self, data, edge_index=None):
        if not edge_index:
            edge_index = data.edge_index
            
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
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

    def forward(self, data, edge_index=[], edge_weight=[]):
        if len(edge_index) == 0:
            edge_index = data.edge_index
        if len(edge_weight) == 0:
            edge_weight = data.edge_weight
        
        x = data.x
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

    def gen_graph_feats(self, G, x):
        dims = torch.tensor(
            [[G.degree(n) for n in range(x.size()[0])]], 
            dtype=torch.float
        ).T
        
        # Normalize and return 
        return dims / dims.max()

    def resample(self, indices, y, max_class, undersample=False):
        # Resample classes at least twice as small as the largest class
        # and sample |MAX_CLASS| / MIN_SAMPLE more values
        print('Resampling')
        MIN_RESAMPLE = 1
        AMT_RESAMPLE = 1
        
        # Split indexes into their classes
        classes = {}
        for i in indices:
            cl = y[i].item()
            if cl in classes:
                classes[cl].append(i)
            else:
                classes[cl] = [i]
                
        print("Has %d classes" % len(classes))
                
        # Find number of samples in largest class
        n = max([len(c) for c in classes.values()])
        for c in classes.keys():
            if len(classes[c]) < n/MIN_RESAMPLE:
                classes[c] = resample(classes[c], n_samples=n//AMT_RESAMPLE)
            
            # Additionally, if we want to undersample the majority class
            # we do that as well
            elif undersample and len(classes[c]) > n/MIN_RESAMPLE:
                classes[c] = resample(
                    classes[c], 
                    n_samples=n//MIN_RESAMPLE,
                    replace=False 
                )
                
        # Put all of the newly sampled arrays together
        ret = []
        for ids in classes.values():
            ret += ids
            
        return np.array(ret)
        

    def generate_pyg_data(self, data): 
        G = nx.Graph()
        has_feats = True
        ADD_GRAPH_FEATS = True

        # Load Feature Table
        x = data['fea_table']
        print("Num Nodes: %d" % x.shape[0])
        print("Node Features: %d" % (int(x.shape[1])-1))

        # Build networkX graph for graph features
        for nid in range(x.shape[0]):
            G.add_node(nid)

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
        
        # Now we can finish building our NetworkX graph and compute graph features
        if ADD_GRAPH_FEATS:
            print("Adding graph feats")
            for [ a, b] in edge_index:
                G.add_edge(a,b) 
            graph_feats = self.gen_graph_feats(G, x)
        
        # Load the graph data
        print('Sorting edges')
        
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        print("Num edges: %d" % edge_index.size()[1])
        
        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
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
        
        '''
        # Let's build a class weight tensor for weighted loss
        weight_vector = [ max(per_class) / a for a in per_class ] 
        print("Weight vector for weighted loss calcs:")
        print(weight_vector)
        '''
        
        # Use beta-normalized weights for the classes
        '''
        beta_sensitivity = int(
            int(log(max(per_class_percent), 8)) - 
            int(log(min(per_class_percent), 8))
        ) + 1
        '''
        
        beta_sensitivity = 2
        
        print('Beta sensitivity: %d' % beta_sensitivity)
        
        BETA = 1-(10 ** -beta_sensitivity)
        per_class = np.array(per_class)
        effective_num = 1.0 - np.power(BETA, per_class)
        weights = (1.0 - BETA) / effective_num
        weight_vector = weights / np.sum(weights) * len(per_class)
        
        '''
        # Use Balanced weighting from Logistic Regression in Rare Events Data, King, Zen, 2001
        per_class = np.array(per_class)
        weight_vector = len(inds) / (len(per_class) * per_class)
        '''
        
        print("Weight vector for weighted loss calcs:")
        print(weight_vector)
        
        # Build train,validate, and test masks
        all_train_indices = data['train_indices']
        print("Num all training: %d" % len(all_train_indices))
        
        '''
        # 80/20 train/val split
        num_train = int(len(all_train_indices) * 0.8)
        num_val = len(all_train_indices) - num_train

        print("Num train nodes: %d" % num_train)
        print("Num val nodes: %d" % num_val)  
        np.random.shuffle(all_train_indices) # make sure indices are random

        train_indices = all_train_indices[:num_train]
        val_indices = all_train_indices[num_train:]
        '''
        
        # More intelligent split that keeps all classes in both splits,
        # and keeps the same distr of classes
        train_indices, val_indices = train_test_split(
            all_train_indices, 
            test_size=0.25,
            stratify=y[inds].numpy(),
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

    def train(self, data):        
        ADD_N2V = False
        ADD_GRAPH_FEATS = True
        
        # Graph data
        avg_degree = degree(data.edge_index[0], data.x.size()[0]).mean()
        
        # Add n2v embeddings to features if there are an order of magnitude more
        # edges than there are features
        if int(log(data.x.size()[0], 10)) < int(log(data.edge_index.size()[1], 10)):
            ADD_N2V = True
        
        # Hyperparamters
        train_epochs=1000
        num_layers=2 # gcn layers
        hidden = min([int(max(data.y)) ** 2, 128])
        early_stopping=True
        val_patience = 100 # how long validation loss can increase before we stop
        
        simplified = True if data.edge_index.size()[1] > 1e6 else False
        simplified = False
        
        print('Hidden dimensions: %d' % hidden)
        if not data.has_features or ADD_N2V or simplified:
            # Requires at least len(class) dimensions, but give it a little more
            embedding_dim = 128 + int(avg_degree ** (1/2))
            
            # The larger the avg degree, the less distant walks matter
            # Of course, a minimum is still important
            context_size = int(log(data.edge_index.size()[1])/avg_degree)
            context_size = context_size if context_size > 2 else 3
            
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
                data, embedder, lr=0.05, patience=3#50 # lower patience when time is important
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
        
        else:
            print('Num feature before: %d' % data.x.size()[1])
            data.x = self.var_thresh(data.x)
            print('Num features after: %d' % data.x.size()[1])

        if ADD_GRAPH_FEATS:
            print('Num feature before: %d' % data.x.size()[1])
            data.x = torch.cat((data.x, data.graph_data), axis=1)
            print('Num features after: %d' % data.x.size()[1])
           
        if simplified:
            print("Using NN")
            model = JustFeatures(
                features_num=data.x.size()[1],
                num_class=int(max(data.y)) + 1, 
                hidden=hidden, 
                num_layers=num_layers
            )
             
        elif data.has_features:
            print("Using GCN")
            model = GCN(
                features_num=data.x.size()[1],
                num_class=int(max(data.y)) + 1, 
                hidden=hidden, 
                num_layers=num_layers
            )
          
        else:
            print("Using GAT")
            heads = int(log(max(data.y) + 1) ** 2)
            print("Num heads: %d" % heads)
            model = GAT(
                features_num=data.x.size()[1], 
                num_class=int(max(data.y)) + 1, 
                #hidden=hidden, 
                num_layers=num_layers,
                heads=heads
            )

        # Move data to compute device
        model = model.to(self.device)
        data = data.to(self.device)
        
        sampler = NeighborSampler(
            data, 
            size=0.75,      # Fraction of neighbors to sample
            num_hops=3,     # How many hops to sample (maybe make a fraction of avg dim?)
            batch_size=16,
            shuffle=True,
            bipartite=False, # Returns a Data object instead of a DataFlow object
        )

        # Configure optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
  
        # Main training loop
        min_loss = float('inf')
        val_loss_min = 1000
        val_increase=0
        stopped_early=False
        state_dict_save = 'checkpoint.model'
             
        for epoch in range(1,train_epochs+1):
            for ns_data in sampler():                
                model.train()
                optimizer.zero_grad()
                loss = F.nll_loss(
                    model(
                        data, 
                        edge_index=ns_data.edge_index,
                        edge_weight=data.edge_weight[ns_data.e_id]
                    )[data.train_mask], 
                    data.y[data.train_mask], 
                    weight=data.weighted_loss
                )
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                
                # calculate loss on validation set
                model.eval()
                loss = F.nll_loss(
                     model(
                        data, 
                        edge_index=ns_data.edge_index,
                        edge_weight=data.edge_weight[ns_data.e_id]
                    )[data.val_mask], 
                    data.y[data.val_mask],
                    weight=data.weighted_loss
                )
                
                val_loss = loss.item()
                print('[%d] Train loss: %.3f   Val Loss: %.3f' % (epoch, train_loss, val_loss))
                if val_loss > val_loss_min and early_stopping:
                    val_increase+= 1
                else:
                    print("===New Minimum validation loss===")
                    val_loss_min = val_loss
                    val_increase=0
                    torch.save(model.state_dict(), state_dict_save)
                        
                if val_increase > val_patience:
                    print("Early stopping!")
                    stopped_early=True
                    break
                
            if stopped_early:
                print("Reloading best parameters!")
                
                # State dict saved to CPU so have to load from there(?)
                model.load_state_dict(torch.load(state_dict_save))
                break
            
        return model

    def pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred

    def train_predict(self, data, time_budget,n_class,schema):
        data = self.generate_pyg_data(data)
        model = self.train(data)
        pred = self.pred(model, data)

        return pred.cpu().numpy().flatten()

    def training_loop(self, data, opt):
        pass
    
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
        
        while(True):
            model.train()
            optimizer.zero_grad()
            loss = model.loss(data.edge_index)
            loss.backward()
            optimizer.step()
            epoch += 1
            
            if verbosity >= 1:
                print('[%d] Loss: %.3f' % (epoch, loss))
            
            if loss > loss_min and early_stopping:
                increase+= 1
            else:
                if verbosity > 0:
                    print("===New Minimum loss===")
                loss_min = loss
                increase=0
                state_dict_save = copy.deepcopy(model.state_dict())
            if increase > patience:
                print("Early stopping!")
                stopped_early=True
                break
        if stopped_early:
            print("Reloading best parameters!")
            model.load_state_dict(state_dict_save)
            
        return model
    
    def var_thresh(self, x, var=0.0):
        sel = VarianceThreshold(var)
        
        x = torch.tensor(sel.fit_transform(x), dtype=torch.float)
        return x
