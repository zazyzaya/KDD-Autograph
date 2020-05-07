import networkx as nx
import numpy as np
import os 
import pandas as pd 

from tqdm import tqdm 
from joblib import delayed, Parallel
from scipy.stats import norm, geom
from math import e

'''
 Uses the same implimentaton for calculating eccentricity
 as nx, but ignores non connected vertices (and is parallel)
'''
def diameter(G, v=None, sample_size=1, workers=4):
    if sample_size != 1:
        if sample_size > 1:
            raise ValueError('Sample size must be >= 1.0')
        
        L = len(G)
        
        v = np.random.choice(
            range(L),
            size=int(sample_size*L),
            replace=False
        )
    
    pbar = tqdm(desc='Nodes parsed', total=len(v))
    
    def get_shortest_path_and_dim(n):
        length=nx.single_source_shortest_path_length(G,n)
        pbar.update()
        
        dim = len(G[n])
        L = len(length)
        
        if L == 0:
            return -1
        else:            
            return max(length.values())
    
    results = Parallel(n_jobs=workers, prefer='threads')(
        delayed(get_shortest_path_and_dim)(n) 
        for n in G.nbunch_iter(v)
    )
    
    return max(results)

def load_graphs(data_dir):
    graphs = {}
    for d in os.listdir(data_dir):
        print("Loading graph from dataset: %s" % d)
        
        G = nx.Graph()
        # First load nodes.  Some graphs had nodes with no edges so had to do this
        print("Loading nodes...")
        with open('%s/%s/train.data/edge.tsv'%(data_dir,d)) as fp:
            lines = fp.readlines()
            for l in lines[1:]:
                l_split = l.split('\t')
                G.add_node(int(l_split[0]))
        print("Loading edges...")
        with open('%s/%s/train.data/edge.tsv'%(data_dir,d)) as fp:
            lines = fp.readlines()
            for l in lines[1:]:
                l_split = l.split('\t')
                G.add_edge(int(l_split[0]), int(l_split[1]))
        print("Done! Num nodes: %d Num edges: %d" % (len(G.nodes()), len(G.edges())))
        graphs[d] = G
    return graphs 

def load_feat_data(data_dir):
    feats = {}
    for d in os.listdir(data_dir):
        print("Loading feats from dataset: %s" % d)

        df = pd.read_csv(
            '%s/%s/train.data/feature.tsv'%(data_dir,d),
            sep='\t',
            header=0,
            index_col=0
        )
        
        feats[d] = df
    
    return feats


'''
Removes any features with variance lower than 
some threshold based on percentile of variance for
all features
'''
def pre_process_feats(X, percentile=.90):
    # Variance threshold    
    v = X.var()
    print(v[1:].mean())
    print(v[1:].std())
    min_var = np.percentile(v[1:], percentile*100)
    
    cols = X.columns
    
    for c in range(len(X.columns)):
        if cols[c] == 'node_index':
            continue
        if v[c] <= min_var:
            X = X.drop(cols[c], axis='columns')
          
    return X


'''
Removes edges with low weight from edge list 
Assumes edges have exponential distro
'''
import torch.nn.functional as F
def pre_process_edge_weights(ew, percentile=0.99):
    print("Ew mean: %0.3f\t Ew std: %0.3f" % (ew.mean(), ew.std()))
    print("Ew max: %0.3f\t Ew min: %0.3f" % (ew.max(), ew.min()))
    
    #min_ew = ew.mean()+ew.std()*geom(1/e).ppf(percentile)*neg
    min_ew = np.percentile(ew.numpy(), percentile*100)
    
    print("Min ew for inclusion: %0.3f" % min_ew)
    ew = ew - min_ew
    
    # Any values less than percentile are now negative.
    # Run through ReLU to turn into 0s 
    return F.relu(ew)

import torch
def get_dim_list(ei, ew, num_nodes):
    el = ei[0].numpy()
    idx, cnt = np.unique(el, return_counts=True)
    cnt = cnt.astype('float64')
    
    dims = torch.zeros((num_nodes,1))
    avg_weight = torch.zeros((num_nodes, 1))
    
    # I don't know if there's a builtin torch method
    # to do this efficiently, but this is the best 
    # I could do...
    for i in range(len(idx)):
        dims[idx[i]] = cnt[i]
   
    curnode = 0     
    weights = []
    for i in range(len(el)):
        if el[i] == curnode:
            weights.append(ew[i])
        else:
            avg_weight[curnode] += sum(weights)/len(weights)
            curnode = el[i]
            weights = [ew[i]]
    avg_weight[curnode] += sum(weights)/len(weights)
    
        
    return torch.cat((dims/dims.max(), avg_weight), dim=1)

def add_dim_to_features(ei, ew, feats):
    dims = get_dim_list(ei, ew, feats.size()[0])
    return torch.cat((feats, dims), dim=1)