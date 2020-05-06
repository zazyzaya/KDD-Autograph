'''
Modules for the models to use. May be combined or tweaked
in any way
'''

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge, Node2Vec
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from math import log 

'''
The original GCN module 
'''
class GCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16,  features_num=16, num_class=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16,  features_num=16, num_class=2):
        super().__init__()
        self.sage1 = SAGEConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.sage1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.sage1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


'''
Adds node2vec embeddings to node features (concatenates them)
'''
class GCN_Plus_N2V(GCN):
    def __init__(self, num_nodes, embedding_dim=16, walk_length=5, context_size=5, 
                 walks_per_node=1, num_layers=2, hidden=32,  features_num=16, num_class=2):
        super().__init__(num_layers, hidden, features_num+embedding_dim, num_class)
        
        self.n2v = Node2Vec(
            num_nodes, 
            embedding_dim, 
            walk_length, 
            context_size, 
            walks_per_node=walks_per_node
        )
        
    def train_n2v(self, data, epochs=100, opt=torch.optim.Adam, 
                  lr=0.01, min_loss=0):
        optimizer = opt(self.n2v.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.n2v.train()
            optimizer.zero_grad()
            loss = self.n2v.loss(data.edge_index)
            loss.backward()
            optimizer.step()
            
            #print("[%d] loss: %.3f" %(epoch, loss.item()))
            if loss.item() <= min_loss:
                break
        
    def reset_parameters(self):
        super().reset_parameters()
        self.n2v.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        embeddings = self.n2v.embedding.weight
        x = torch.cat((x, embeddings), dim=1)
        
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


'''
Uses node2vec embeddings and features, but does not use a GCN, 
just treats them as vectors with no other information
'''
class N2V_Plus_Features(GCN_Plus_N2V):
    def __init__(self, num_nodes, embedding_dim=16, walk_length=5, context_size=5, 
                 walks_per_node=1, num_layers=2, hidden=32, features_num=16, num_class=2):
        
        super().__init__(num_nodes, embedding_dim, walk_length, context_size, 
                 walks_per_node, num_layers, hidden,  features_num, num_class)
        
        self.first_lin = Linear(features_num+embedding_dim, hidden)
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(Linear(hidden, hidden))
        
        self.last_lin = Linear(hidden, num_class)

    def reset_parameters(self):
        self.n2v.reset_parameters()
        self.first_lin.reset_parameters()
        for l in self.hidden_layers:
            l.reset_parameters()
        self.last_lin.reset_parameters()
            
    def forward(self, data):
        x = data.x 
        x = torch.cat((x, self.n2v.embedding.weight), dim=1)
        
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for l in self.hidden_layers:
            x = F.relu(l(x))
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_lin(x)
        return F.log_softmax(x, dim=-1)
    

class JustFeaturesModule(torch.nn.Module):
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