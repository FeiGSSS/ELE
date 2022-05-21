# -*- encoding: utf-8 -*-
'''
@File    :   MC.py
@Time    :   2022/04/28 18:24:22
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import random
import numpy as np

import time
from numba import njit
from utils import convert2nb_dict, ifConverge
from multiprocessing import Pool


@njit()
def iter(rho, tmax, I, N, gamma, beta, beta_D, node_neighbors_dict, tri_neighbors_dict):
    for t in range(1, tmax):
        if np.sum(I) < 1e-4 or abs(np.sum(I)-N)<1:
            for t2 in range(t, tmax):
                rho[t2] = np.mean(I)
            return rho
        
        I_new = I.copy()
        for node in range(N):
            if I[node] == 1:  # infected
                if random.random() < gamma:
                    I_new[node] = 0
            else:
                for v in node_neighbors_dict[node]:
                    if I[v] == 1 and random.random() < beta:
                        I_new[node] = 1
                        break
                
                if I_new[node] == 1:
                    continue

                try:
                    neighbors = tri_neighbors_dict[node]
                    for (u, v) in neighbors:
                        if I[u] == 1 and I[v] == 1 and random.random() < beta_D:
                            I_new[node] = 1
                            break
                except:
                    pass
            
        I = I_new.copy()
        rho[t] = np.mean(I)
        if ifConverge(rho[:t], N):
            rho = rho[:t]
            return rho
    return rho



def Hror_SIS_MC_once(seed, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict, tmax, I0, steady):
    random.seed(seed)
    np.random.seed(seed)

    N = len(node_neighbors_dict)
    if isinstance(I0, float):
        I0 = np.random.choice(range(N), int(N*I0), replace=False)
    else:
        I0 = np.array(I0, dtype=np.int64)

    # init
    I = np.zeros(N)
    I[I0] = 1
    
    rho = np.zeros(tmax, dtype=np.float64)
    rho[0] = np.mean(I)
    rho = iter(rho, tmax, I, N, gamma, beta, beta_D, node_neighbors_dict, tri_neighbors_dict)

    if not steady:
        out = rho
    else:
        out = np.mean(rho[-100:])

    return out

def Hror_SIS_MC_batch(batch_size, seed, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict, tmax, I0, steady):
    
    node_neighbors_dict, tri_neighbors_dict  = convert2nb_dict(node_neighbors_dict, tri_neighbors_dict)
    rhos = []
    for b in range(batch_size):
        bseed = seed + b + int(time.time() * 1000) % 1000
        rhos.append(Hror_SIS_MC_once(bseed, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict, tmax, I0, steady))
    
    return rhos

def Hror_SIS_MC(beta, 
                beta_D, 
                gamma,
                node_neighbors_dict, 
                tri_neighbors_dict, 
                tmax,
                I0, 
                pool_num, 
                iteration, 
                steady=True):

    batch_size = int(iteration / pool_num)
    
    pool = Pool(pool_num)
    results = []
    for p in range(pool_num):
        results.append(pool.apply_async(Hror_SIS_MC_batch, (batch_size, p, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict, tmax, I0, steady)))
    pool.close()
    pool.join()

    rhos = []
    for r in results:
        rhos.extend(r.get())
    
    return rhos