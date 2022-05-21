from email import header


# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/05/21 16:26:23
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
from numba.typed import Dict 
from numba import types, njit

def convert2nb_dict(node_neighbors_dict, tri_neighbors_dict):
    node_neighbors_dict_nb = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    tri_neighbors_dict_nb  = Dict.empty(key_type=types.int64, value_type=types.int64[:,:])
    for k, v in node_neighbors_dict.items():
        if len(v) == 0:
            continue
        else:
            node_neighbors_dict_nb[int(k)] = np.array(v)

    for k,v in tri_neighbors_dict.items():
        if len(v) == 0:
            continue
        else:
            tri_neighbors_dict_nb[int(k)] = np.array(list(v))
    return node_neighbors_dict_nb, tri_neighbors_dict_nb


def parser_mc_results(rho, cut:bool=True):
    rhos_array = np.vstack([x[1] for x in sorted(zip(rho.keys(), rho.values()), key=lambda x:x[0])]).T

    if cut:
        cut_point = min(np.argwhere(np.count_nonzero(rhos_array, axis=0)>1))[0]
        cut_rhos_array = []

        for rhos in rhos_array:
            clean_rhos = []
            for i, rr in enumerate(rhos):
                if i<cut_point:
                    clean_rhos.append(rr)
                elif rr==0:
                    clean_rhos.append(np.nan)
                else:
                    clean_rhos.append(rr)
            cut_rhos_array.append(clean_rhos) 

        cut_rhos_array = np.array(cut_rhos_array)
        avg_rhos = np.nanmean(cut_rhos_array, axis=0)
    else:
        avg_rhos = np.mean(rhos_array, axis=0)
        
    return avg_rhos

def parser_results(rho):
    return np.array([x[1] for x in sorted(zip(rho.keys(), rho.values()), key=lambda x:x[0])])

@njit()
def ifConverge(rho:np.ndarray, N:int, threshold:float=1e-3)->bool:
    if len(rho) < 100:
        return False
    else:
        std = np.std(rho[-100:]) * N
        if std < threshold:
            return True
        else:
            return False

import os
def checkFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        