# -*- encoding: utf-8 -*-
'''
@File    :   loadate.py
@Time    :   2022/05/21 16:20:19
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
"""
This script produce two kinds of random simplicial complexes networks, ER and SF.
Ref to paper "Full reconstruction of simplicial complexes from binary time-series data"
"""

import copy
import networkx as nx
import numpy as np

def add_triangle_to_graph(triangles:list, graph:nx.graph):
    g = copy.deepcopy(graph)
    for tri in triangles:
        assert len(set(tri)) == 3, print(tri)
        i, j, k = tri
        g.add_edges_from([[i,j], [i,k], [j,k]])
    return g

def post_process(n:int, G:nx.Graph, p2:float):
    num_of_triangles = int(p2*n*(n-1)*(n-2)/6)
    # BUG>>> duplicated triangles , maybe not a big deal
    triangles = [list(np.random.choice(n, size=3, replace=False)) for _ in range(num_of_triangles)]
    G = add_triangle_to_graph(triangles, G)

    # exclude isolated nodes
    for node in G.nodes():
        if G.degree(node) == 0:
            target = np.random.choice(range(n), 1)
            G.add_edges_from([(node, target[0])])

    # rearange nodes' indexes
    N = G.order()
    node_relabel_map = {node:i for i, node in enumerate(G.nodes())}
    nx.relabel_nodes(G, node_relabel_map)
                
    # Creating a list of neighbors, default as empty list
    node_neighbors_list = [[] for _ in range(N)]
    for n in G.nodes():
        node_neighbors_list[n] = list(G.neighbors(n))

    triangles_list = [[node_relabel_map[node] for node in  tri] for tri in triangles]
    tri_neighbors_list = [set() for _ in range(N)]
    for (i, j, k) in triangles_list:
        tri_neighbors_list[i].add((j,k))
        tri_neighbors_list[j].add((i,k))
        tri_neighbors_list[k].add((i,j))

    node_neighbors_dict = {node:value for node, value in enumerate(node_neighbors_list)}
    tri_neighbors_dict = {node:value for node, value in enumerate(tri_neighbors_list)}

    avg_k1 = sum([len(neighbors) for neighbors in node_neighbors_dict.values()])/N
    avg_k2 = sum([len(neighbors) for neighbors in tri_neighbors_dict.values()])/N 

    return node_neighbors_dict, tri_neighbors_dict, avg_k1, avg_k2

def import_random_ersc(n:int=200, k1:float=6.0, k2:float=2.0):
    p1 = (k1-2*k2) / (n-1-2*k2)
    p2 = 2*k2 / ((n-1)*(n-2))
    G = nx.erdos_renyi_graph(n, p1)

    return post_process(n, G, p2)

def import_random_sfsc(n:int, k1:float, k2:float):
    m = max(1, int(n*(k1-2*k2) / (2*n - 4*k2)))
    p2 = 2*k2/((n-1)*(n-2))
    G = nx.barabasi_albert_graph(n=n, m=m)

    return post_process(n, G, p2)
    
