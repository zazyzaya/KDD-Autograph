'''
Dataset,score,      time
A,      0.8716,     16.120949029922485     
B,      0.7486,     4.425185918807983
C,      0.8421,     57.29339838027954
D,      0.9306,     167.68811798095703
E,      0.8353,     98.71457004547119
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

    def gen_graph_feats(self, G):
        degree_list = [ G.degree(n) for n in G.nodes() ]
        ccs = list(nx.connected_components(G))
        node_to_cc_size = {}
   
        max_cc = []
        
        for cc in ccs:
            curr_cc_size = len(cc)
            if curr_cc_size > len(max_cc):
                max_cc = cc
            for nid in cc:
                node_to_cc_size[nid] = curr_cc_size
 
        graph_feats = []
        for n in range(len(G.nodes())):
            #graph_feats.append([ G.degree(n), node_to_cc_size[n] ])
            graph_feats.append([ G.degree(n) ])

        graph_feats = np.array(graph_feats)
        #for column in range(graph_feats.shape[1]):
        #    c_smallest = graph_feats[:,column].min()
            # shift smallest to 0
        #    graph_feats[:,column] = graph_feats[:,column] - c_smallest

            # scale max to 1
        #    c_largest = graph_feats[:,column].max()
        #    print("Column %d largest: %d" % (column, int(c_largest)))
        #    graph_feats[:,column] = graph_feats[:,column] / c_largest
        return graph_feats, max_cc

    def generate_pyg_data(self, data):
        NORMALIZE_FEATS = False
        ADD_GRAPH_FEATS = False
        TRAIN_MAX_CC_ONLY = False 

        G = nx.Graph()
        has_feats = True

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
            if (x.max() > 1.0 or x.min() < 0.0) and NORMALIZE_FEATS:
                print("Normalizing features by column into [0,1] ...")
                for column in range(x.shape[1]):
                   c_smallest = x[:,column].min()
                   # shift smallest to 0
                   x[:,column] = x[:,column] + abs(c_smallest)
                 
                   # scale max to 1
                   c_largest = x[:,column].max()
                   x[:,column] = x[:,column] / c_largest
                print("Feature normalization complete:")
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
        
        # Now we can finish building our NetworkX graph and compute graph features
        if ADD_GRAPH_FEATS and has_feats:
            print("Adding graph feats")
            for [ a, b] in edge_index:
                G.add_edge(a,b) 
            graph_feats, max_cc = self.gen_graph_feats(G)
            x = np.hstack((x, graph_feats))
        
        # Convert input data to tensor
        x = torch.tensor(x, dtype=torch.float)

        # Load the graph data
        print('Sorting edges')
        
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        
        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # This is a very computationally expensive line of code.
        # print("Max/Min edge weight: %f/%f" % (max(edge_weight), min(edge_weight)))

        # Build train,validate, and test masks
        all_train_indices = data['train_indices']
        print("Num all training: %d" % len(all_train_indices))
       
        if TRAIN_MAX_CC_ONLY and has_feats:
            # Only include those from maximum connected component
            max_cc_only_indices = set(all_train_indices).intersection(set(max_cc))
            print("Max cc only training indices: %d" % len(max_cc_only_indices))
            all_train_indices = list(max_cc_only_indices)

        # 80/20 train/val split
        num_train = int(len(all_train_indices) * 0.8)
        num_val = len(all_train_indices) - num_train

        print("Num train nodes: %d" % num_train)
        print("Num val nodes: %d" % num_val)  
        np.random.shuffle(all_train_indices) # make sure indices are random

        train_indices = all_train_indices[:num_train]
        val_indices = all_train_indices[num_train:]
        
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
        # Let's build a class weight tensor for weighted loss
        weight_vector = [ max(per_class) / a for a in per_class ] 
        print("Weight vector for weighted loss calcs:")
        print(weight_vector)

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
        data.weighted_loss = torch.tensor(weight_vector)

        return data

    def train(self, data):        
        ADD_N2V = False
        
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
        val_patience = 50 # how long validation loss can increase before we stop

        # If there are too many edges for GCN, use n2v+features
        simplified = True if (data.edge_index.size()[1] > 1e6) else False
        
        print(hidden)
        if not data.has_features or simplified or ADD_N2V:
            # Requires at least len(class) dimensions, but give it a little more
            embedding_dim = 128 + int(data.weighted_loss.size()[0] ** (1/2))
            
            # The larger the avg degree, the less distant walks matter
            # Of course, a minimum is still important
            context_size = int(log(data.edge_index.size()[1])/avg_degree)
            context_size = context_size if context_size > 2 else 3
            
            # We should look at at least 1 context per walk
            walk_len = context_size*2 + 1
        
            print('Embedding dim: %d\tWalk Len: %d\tContext size: %d'
                  % (embedding_dim, walk_len, context_size))
        
            embedder = Node2Vec(
                data.x.size()[0],   # Num nodes
                embedding_dim,      # Embedding dimesion
                walk_len,           # Walk len  
                context_size       # Context size 
            )
            
            # First, train embedder
            # Use a higher learning rate, bc this part is
            # meant to be kind of "quick and dirty"
            embedder = self.n2v_trainer(
                data, embedder, lr=0.05#, patience=100 # lower patience when time is important
            )
            
            if (simplified and data.has_features) or ADD_N2V:
                data.x = torch.cat((var_thresh(data.x), embedder.embedding.weight), axis=1)
            else:
                # Then use n2v embeddings as features
                data.x = embedder.embedding.weight
        
        else:
            print('Num feature before: %d' % data.x.size()[1])
            data.x = var_thresh(data.x)
            print('Num features after: %d' % data.x.size()[1])

        if simplified:
            model = JustFeatures(
                features_num=data.x.size()[1], 
                num_class=int(max(data.y)) + 1, 
                hidden=hidden, 
                num_layers=num_layers
            )
            
        else:
            model = GCN(
                features_num=data.x.size()[1], 
                num_class=int(max(data.y)) + 1, 
                hidden=hidden, 
                num_layers=num_layers
            )

        # Move data to compute device
        model = model.to(self.device)
        data = data.to(self.device)

        # Configure optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
  
        # Main training loop
        min_loss = float('inf')
        val_loss_min = 1000
        val_increase=0
        stopped_early=False
        state_dict_save = {}
        for epoch in range(1,train_epochs+1):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask], weight=data.weighted_loss)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            # calculate loss on validation set
            model.eval()
            loss = F.nll_loss(model(data)[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            print('[%d] Train loss: %.3f   Val Loss: %.3f' % (epoch, train_loss, val_loss))
            if val_loss > val_loss_min and early_stopping:
                val_increase+= 1
            else:
                print("===New Minimum validation loss===")
                val_loss_min = val_loss
                val_increase=0
                state_dict_save = copy.deepcopy(model.state_dict())
            if val_increase > val_patience:
                print("Early stopping!")
                stopped_early=True
                break
        if stopped_early:
            print("Reloading best parameters!")
            model.load_state_dict(state_dict_save)
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

    def n2v_trainer(self, data, model, epochs=800, early_stopping=True, 
                    patience=10, verbosity=1, lr=0.01):
        
        print("Training n2v")
        model = model.to(self.device)
        data = data.to(self.device)
        
        loss_min=1000
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        increase = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = model.loss(data.edge_index)
            loss.backward()
            optimizer.step()
            
            if verbosity >= 1:
                print('[%d] Loss: %.3f' % (epoch, loss))
            
            if loss > loss_min and early_stopping:
                increase+= 1
            else:
                if verbosity > 0:
                    print("===New Minimum validation loss===")
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
    
from sklearn.feature_selection import VarianceThreshold
def var_thresh(x, var=0.0):
    sel = VarianceThreshold(var)
    
    x = torch.tensor(sel.fit_transform(x), dtype=torch.float)
    return x