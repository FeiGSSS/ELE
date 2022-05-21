from email import header


# -*- encoding: utf-8 -*-
'''
@File    :   mmca.py
@Time    :   2022/04/05 19:06:50
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
from utils import convert2nb_dict, ifConverge
from numba import njit

@njit()
def update(N, I, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict):
    Inew = I.copy()
    for i in range(N):
        qi = 1
        try:
            for j in node_neighbors_dict[i]:
                qi *= 1 - beta * I[j]
        except:
            pass

        try:
            for (j, k) in tri_neighbors_dict[i]:
                qi *= 1 - beta_D * I[j] * I[k]
        except:
            pass
            
        Inew[i] = (1 - I[i]) * (1 - qi) + I[i] * (1 - gamma)
    return Inew


def Hror_SIS_MMCA(beta, 
                  beta_D, 
                  gamma,
                  node_neighbors_dict, 
                  tri_neighbors_dict,
                  tmax, 
                  I0,
                  steady:bool=True):
    node_neighbors_dict, tri_neighbors_dict = convert2nb_dict(node_neighbors_dict, tri_neighbors_dict)
    N = len(node_neighbors_dict)
    I = np.zeros(N)
    for idx in I0:
        I[idx] = 1
    rho = np.zeros(tmax)         
    rho[0] = np.mean(I)
    for t in range(1, tmax):
        I = update(N, I, beta, beta_D, gamma, node_neighbors_dict, tri_neighbors_dict)
        rho[t] = np.mean(I)
        if ifConverge(rho[:t], N):
            rho = rho[:t]
            break
    
    if steady:
        out = np.mean(rho[-100:])
    else:
        out = rho
    
    return beta, out