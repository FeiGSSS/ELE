# -*- encoding: utf-8 -*-
'''
@File    :   ELE.py
@Time    :   2022/04/19 16:29:14
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from utils import convert2nb_dict, ifConverge

@njit
def update_P(PI, PII, PSI, PSS):
    for (i,j), _ in PII.items():
        PSI[(i,j)] = PI[j] - PII[(i,j)]
        PSI[(j,i)] = PI[i] - PII[(i,j)]
        PSS[(i,j)] = 1 -  PSI[(j,i)] - PSI[(i,j)] - PII[(i,j)]

@njit
def update_PSII(PI, PSI, PII, tri_neighbors_dict, PSII):
    for i, nei in tri_neighbors_dict.items():
        for (j, r) in nei:
            j, r = sorted([j, r])
            condition = (1-PI[i]) * PI[j] * PI[r]
            if condition == 0:
                PSII[(i,j,r)] = 0
            else:
                PSII[(i,j,r)] = (PSI[(i,j)] * PSI[(i,r)] * PII[(j,r)]) / condition

@njit
def update_qi_qiD(N, PI, PSI, PSII, beta, beta_D, node_neighbors_dict, tri_neighbors_dict, node_keys, tri_keys, qi, qi_D):
    for i in range(N):
        if PI[i] == 1:
            qi[i] = 1
            qi_D[i] = 1
        else:
            condition = 1-PI[i]
            # qi
            if i in node_keys:
                value = 1
                for j in node_neighbors_dict[i]:
                    value *= (1 - beta * PSI[(i,j)] / condition)
                qi[i] = value
            else:
                qi[i] = 1
            
            # qi_D
            if i in tri_keys:
                value = 1
                for (j,r) in tri_neighbors_dict[i]:
                    j, r = sorted([j ,r])
                    value *= (1 - beta_D * PSII[(i,j,r)] / condition)
                qi_D[i] = value
            else:
                qi_D[i] = 1

@njit
def update_qij_qijD_uij(PI, PSI, PSII, beta, beta_D, qi, qi_D, tri_neighbors_dict, tri_keys, qij, qij_D, uij):
    for (i, j) in PSI.keys():
        if 1 == PI[i]:
            qij[(i, j)] = 1
            qij_D[(i,j)] = 1
            uij[(i, j)] = 1

        else:
            condition = 1 - PI[i]
            qij[(i, j)] = qi[i] / (1 - beta * PSI[(i,j)] / condition)
            
            qij_D[(i,j)] = qi_D[i]
            if i in tri_keys:
                exception_D = 1
                for (r, l) in tri_neighbors_dict[i]:
                    if j in [r,l]:
                        r, l = sorted([r, l])
                        exception_D *=  1 - beta_D * PSII[(i,r,l)] / condition
                qij_D[(i,j)] /= exception_D
            else:
                pass
        
        if PSI[(i,j)] == 0:
            uij[(i, j)] = 1
        else:
            uij[(i, j)] = 1
            if i in tri_keys:
                for (r, l) in tri_neighbors_dict[i]:
                    if j in [r,l]:
                        r, l = sorted([r, l])
                        uij[(i, j)] *= 1 - beta_D * PSII[(i,r,l)] / PSI[(i,j)]
            else:
                pass

@njit
def update_PI(N, PI, gamma, qi, qi_D, PI_new):
    for i in range(N):
        PI_new[i] = (1 - PI[i]) * (1 - qi[i] * qi_D[i]) + PI[i] * (1 - gamma)

@njit
def update_PII(PSS, PSI, PII, qij, qij_D, uij, beta, gamma, PII_new):
    for (i, j) in PII.keys():
        part1 = PSS[(i, j)] * (1 - qij[(i,j)] * qij_D[(i, j)]) * (1 - qij[(j,i)] * qij_D[(j, i)])
        part2 = PSI[(i, j)] * (1 - (1 - beta) * qij[(i, j)] * uij[(i,j)] * qij_D[(i,j)]) * (1 - gamma)
        part3 = PSI[(j, i)] * (1 - (1 - beta) * qij[(j, i)] * uij[(j,i)] * qij_D[(j,i)]) * (1 - gamma)
        part4 = PII[(i, j)] * (1 - gamma)**2
        PII_new[(i, j)] = part1 + part2 + part3 + part4

def _Hror_dSIS_ELE_base(N,
                        gamma, 
                        beta, 
                        beta_D,
                        PI, 
                        PII,
                        node_neighbors_dict, 
                        tri_neighbors_dict,
                        node_keys,
                        tri_keys):

    PSI = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    PSS = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_P(PI, PII, PSI, PSS)

    PSII = Dict.empty(key_type=types.UniTuple(types.int64, 3), value_type=types.float64)
    update_PSII(PI, PSI, PII, tri_neighbors_dict, PSII)

    qi = Dict.empty(key_type=types.int64, value_type=types.float64)
    qi_D = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_qi_qiD(N, PI, PSI, PSII, beta, beta_D, node_neighbors_dict, tri_neighbors_dict, node_keys, tri_keys, qi, qi_D)
    

    qij = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    qij_D = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    uij = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_qij_qijD_uij(PI, PSI, PSII, beta, beta_D, qi, qi_D, tri_neighbors_dict, tri_keys, qij, qij_D, uij)

    PI_new = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_PI(N, PI, gamma, qi, qi_D, PI_new)

    PII_new = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_PII(PSS, PSI, PII, qij, qij_D, uij, beta, gamma, PII_new)
    
    return PI_new, PII_new


def Hror_SIS_ELE(beta,
                 beta_D,
                 gamma,
                 node_neighbors_dict,
                 tri_neighbors_dict,
                 tmax,
                 I0,
                 steady=True):
    N = len(node_neighbors_dict)
    node_neighbors_dict, tri_neighbors_dict = convert2nb_dict(node_neighbors_dict, tri_neighbors_dict)
    node_keys = np.array(list(node_neighbors_dict.keys()))
    tri_keys  = np.array(list(tri_neighbors_dict.keys()))

    PI = Dict.empty(key_type=types.int64, value_type=types.float64)
    for i in range(N):
        PI[i] = 1 if i in I0 else 0
    
    PII = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    for i in range(N):
        for j in node_neighbors_dict[i]:
            i1, j1 = sorted([i,j])
            if i1 in I0 and j1 in I0:
                PII[(i1,j1)] = 1
            else:
                PII[(i1,j1)] = 0
    
    rho = np.zeros(tmax)
    rho[0] = sum(PI.values()) / N
    for t in range(1, tmax):
        PI, PII = _Hror_dSIS_ELE_base(N,
                                    gamma,
                                    beta,
                                    beta_D,
                                    PI,
                                    PII,
                                    node_neighbors_dict,
                                    tri_neighbors_dict,
                                    node_keys,
                                    tri_keys)
        rho[t] = sum(PI.values()) / N
        if ifConverge(rho[:t], N):
            rho = rho[:t]
            break
    print("=", end="")
    if steady:
        return beta, np.mean(rho[-100:])
    else:
        return beta, rho