import networkx as nx
import numpy as np
from joblib import delayed, Parallel

'''
 Uses the same implimentaton for calculating eccentricity
 as nx, but ignores non connected vertices (and is parallel)
'''
def graph_data(G, v=None, sample_size=1, workers=4):
    if sample_size != 1:
        if sample_size > 1:
            raise ValueError('Sample size must be >= 1.0')
        
        L = len(G)
        
        v = np.random.choice(
            range(L),
            size=int(sample_size*L),
            replace=False
        )
    
    def get_shortest_path_and_dim(n):
        length=nx.single_source_shortest_path_length(G,n)
        dim = len(G[n])
        L = len(length)
        
        if L == 0:
            return [-1,dim]
        else:            
            return [max(length.values()),dim]
    
    results = Parallel(n_jobs=workers, prefer='threads')(
        delayed(get_shortest_path_and_dim)(n) 
        for n in G.nbunch_iter(v)
    )
    
    mp = 0
    md = 0
    for r in results:
        if r[0]>mp: 
            mp=r[0]
        if r[1]>md:
            md=r[1]
            
    return mp, md