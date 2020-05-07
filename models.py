'''
Lots of custom models to work on the graph problems. 
Import any into model.py as Model for run_local to work
'''

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, Node2Vec
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from graph_utils import *
from math import log 

from modules import * 

'''
The original model 
'''
class OGModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def generate_pyg_data(self, data):
        x = data['fea_table']
        x = pre_process_feats(x)

        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
            has_features = False
        else:
            x = x.drop('node_index', axis=1).to_numpy()
            has_features = True
        
        x = torch.tensor(x, dtype=torch.float)
        x = add_dim_to_features(edge_index, edge_weight, x)
        
        print(edge_weight)
        # For weighted graphs, only keep edges with high weights
        if edge_weight.max() > 1:
            print("Pruning low-weight edges")
            print("Num edges before: %d" % len(edge_weight))
            edge_weight = pre_process_edge_weights(edge_weight)
            edge_index = edge_index[:, edge_weight != 0]
            edge_weight = edge_weight[edge_weight != 0]
            print("Num edges after: %d" % len(edge_weight))

        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        # Graph metadata
        data.has_features = has_features
        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        
        # Include class weights for better optim
        num_classes=int(max(y)) + 1 
        per_class = [ 0 ] * num_classes
        for label in y[inds]:
            per_class[label] += 1
        weight_vector = [ max(per_class) / a for a in per_class ] 
        data.class_weights = torch.tensor(weight_vector)
        
        print(data.x)
        print("Num features: %d" % data.x.size()[1])
        return data

    
    def train(self, data):
        model = GCN(
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1,
            aggr='mean'
        )
        
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
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

    
'''
Generates node2vec embeddings, then adds them to node features. 
Carries on as normal after that
'''
class Node2VecCombo(OGModel):
    def __init__(self):
        super().__init__()
    
    def train(self, data):
        model = GCN_Plus_N2V(
            data.x.size()[0], # Num nodes
            embedding_dim=16,
            walks_per_node=100,
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1,
        )
        
        model = model.to(self.device)
        data = data.to(self.device)
        
        model.train_n2v(data, lr=0.1, min_loss=0.9)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        return model
    
    
'''
Generates node2vec embeddings, then completely ignores graph
structure from there on out. Just cats them to node feats 
and runs them through standard classifier
'''
class Node2VecFeatures(OGModel):
    def __init__(self):
        super().__init__()
    
    def train(self, data):
        model = N2V_Plus_Features(
            data.x.size()[0], # Num nodes
            embedding_dim=16,
            walks_per_node=100,
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1,
        )
        
        model = model.to(self.device)
        data = data.to(self.device)
        
        model.train_n2v(data, lr=0.1, min_loss=0.9)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        return model
    
    
class JustFeatures(OGModel):
    def __init__(self):
        super().__init__()
        
    def train(self, data):
        model = JustFeaturesModule(
            features_num=data.x.size()[1],
            num_class=int(max(data.y))+1,
            num_layers=3,
            hidden=128
        )
        
        print('Num features: %d\tNum classes: %d' % (data.x.size()[1], int(max(data.y))+1))
        print('Num nodes: %d\tNum edges:%d' % (data.x.size()[0], data.edge_index.size()[1]))        
        model = model.to(self.device)
        data = data.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask], weight=data.class_weights)
            loss.backward()
            optimizer.step()
        return model
    
    
class GraphSAGEModel(OGModel):
    def __init__(self):
        super().__init__()
        
    def train(self, data):
        model = GraphSAGE(
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1,
            num_layers=2,
            hidden=16
        )
        
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask], weight=data.class_weights)
            loss.backward()
            optimizer.step()
        return model
    
class Node2VecModel(OGModel):
    def __init__(self):
        super().__init__()
        
    # TODO 
    def train(self, data):
        pass