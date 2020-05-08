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
from torch_geometric.nn import Node2Vec

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
        
        x = normalize_features(x)
        x = torch.tensor(x, dtype=torch.float)
        
        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        # 80/20 train/val split
        all_train_indices = data['train_indices']
        num_train = int(len(all_train_indices) * 0.8)
        
        # make sure indices are random
        np.random.shuffle(all_train_indices) 

        # Split train/val set
        train_indices = all_train_indices[:num_train]
        val_indices = all_train_indices[num_train:]
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        # Graph metadata
        data.has_features = has_features
        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_indices] = 1
        data.val_mask = val_mask

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
        )
        
        return generic_training_loop(data, model, self.device)

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
        
        return generic_training_loop(data, model, self.device)
    
    
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
        
        return generic_training_loop(data, model, self.device)
    
    
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
        
        return generic_training_loop(data, model, self.device)
    
    
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
        
        return generic_training_loop(data, model, self.device)
    
class BenGCNModel(OGModel):
    def __init__(self):
        super().__init__()
        
    def train(self, data):
        if data.has_features == False:
            embedding_dim = 128
        
            embedder = Node2Vec(
                data.x.size()[0],   # Num nodes
                embedding_dim,      # Embedding dimesion
                7,                  # Walk len  
                3,                  # Context size 
            )
            
            # First train embedder
            # Use a higher learning rate, bc this part is
            # meant to be kind of "quick and dirty"
            embedder = n2v_trainer(
                data, embedder, self.device, lr=0.1
            )
            
            # Then use n2v embeddings as features
            data.x = embedder.embedding.weight
        
        model = BenGCN(
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1, 
            num_layers=2
        )
        
        return generic_training_loop(data, model, self.device)

class BenSAGEModel(OGModel):
    def __init__(self):
        super().__init__()
        
    def train(self, data):
        if data.has_features == False:
            embedding_dim = 128
        
            embedder = Node2Vec(
                data.x.size()[0],   # Num nodes
                embedding_dim,      # Embedding dimesion
                7,                  # Walk len  
                3,                  # Context size 
            )
            
            # First train embedder
            # Use a higher learning rate, bc this part is
            # meant to be kind of "quick and dirty"
            embedder = n2v_trainer(
                data, embedder, self.device, lr=0.1
            )
            
            # Then use n2v embeddings as features
            data.x = embedder.embedding.weight
        
        model = BenSAGE(
            features_num=data.x.size()[1], 
            num_class=int(max(data.y)) + 1, 
            num_layers=2
        )
        
        return generic_training_loop(data, model, self.device)


import copy 
def generic_training_loop(data, model, device, epochs=800, early_stopping=True, 
                          val_loss_min=1000, val_patience=50, verbosity=0.5, lr=0.005):
    # calculate loss on validation set
    model = model.to(device)
    data = data.to(device)
    
    val_increase = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in range(1,epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask], weight=data.class_weights)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        loss = F.nll_loss(model(data)[data.val_mask], data.y[data.val_mask])
        val_loss = loss.item()
        if verbosity >= 1:
            print('[%d] Train loss: %.3f   Val Loss: %.3f' % (epoch, train_loss, val_loss))
        if val_loss > val_loss_min and early_stopping:
            val_increase+= 1
        else:
            if verbosity > 0:
                print("===New Minimum validation loss===")
                print("\t\t%0.3f" % val_loss)
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

def n2v_trainer(data, model, device, epochs=800, early_stopping=True, 
                patience=50, verbosity=0.5, lr=0.01):
    
    model = model.to(device)
    data = data.to(device)
    
    loss_min=1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    increase = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data.edge_index)
        loss.backward()
        optimizer.step()
        
        if loss > loss_min and early_stopping:
            increase+= 1
        else:
            if verbosity > 0:
                print("===New Minimum validation loss===")
                print("\t\t%0.3f" % loss)
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