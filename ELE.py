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

@njit()
def update_P(PI, PII, PIS, PSI, PSS):
    for k, _ in PII.items():
        i, j = k
        PIS[k] = PI[i] - PII[k]
        PSI[k] = PI[j] - PII[k]
        PSS[k] = 1 - PIS[k] - PSI[k] - PII[k]

@njit()
def update_PSII(PI, PSI, PII, tri_neighbors_dict, PSII):
    for i, nei in tri_neighbors_dict.items():
        for (j, r) in nei:
            j, r = sorted([j, r])
            condition = (1-PI[i]) * PI[j] * PI[r]
            if condition == 0:
                PSII[(i,j,r)] = 0
            else:
                PSII[(i,j,r)] = (PSI[(i,j)] * PSI[(i,r)] * PII[(j,r)]) / condition

@njit()
def update_qi_qiD(N, PI, PSI, PSII, beta, beta_D, node_neighbors_dict, tri_neighbors_dict, qi, qi_D):
    for i in range(N):
        if PI[i] == 1:
            qi[i] = 1
            qi_D[i] = 1
        else:
            # qi
            try:
                neighbors = node_neighbors_dict[i]
                value = 1
                condition = 1-PI[i]
                for j in neighbors:
                    value *= (1 - beta * PSI[(i,j)] / condition)
                qi[i] = value
            except:
                qi[i] = 1
            
            # qi_D
            try:
                neighbors = tri_neighbors_dict[i]
                value = 1
                condition = 1-PI[i]
                for (j,r) in neighbors:
                    j, r = sorted([j ,r])
                    value *= (1 - beta_D * ( PSII[(i,j,r)] / condition))
                qi_D[i] = value
            except:
                qi_D[i] = 1

@njit()
def update_qij_qijD_uij(edges, PI, PSI, PSII, beta, beta_D, qi, qi_D, tri_neighbors_dict, qij, qij_D, uij):
    for (i, j) in edges:
        if 1 == PI[i]:
            qij[(i, j)] = 1
            qij_D[(i,j)] = 1
            uij[(i, j)] = 1

        else:
            qij[(i, j)] = qi[i] / (1 - beta * PSI[(i,j)] / (1 - PI[i]))

            condition = 1 - PI[i]

            qij_D[(i,j)] = qi_D[i]
            try:
                tri_neighbors = tri_neighbors_dict[i]
                exception_D = 1
                for (r, l) in tri_neighbors:
                    if r == j or l == j:
                        r, l = sorted([r, l])
                        exception_D *=  1 - beta_D * PSII[(i,r,l)] / condition
                qij_D[(i,j)] /= exception_D
            except:
                pass

            uij[(i, j)] = 1
            try:
                tri_neighbors = tri_neighbors_dict[i]
                for (r, l) in tri_neighbors:
                    if r == j or l == j:
                        r, l = sorted([r, l])
                        uij[(i, j)] *= 1 - beta_D * PSII[(i,r,l)] / condition
            except:
                pass

@njit()
def update_PI(N, PI, gamma, qi, qi_D, PI_new):
    for i in range(N):
        PI_new[i] = (1 - PI[i]) * (1 - qi[i] * qi_D[i]) + PI[i] * (1 - gamma)

@njit()
def update_PII(edges, PSS, PSI, PIS, PII, qij, qij_D, uij, beta, gamma, PII_new):
    for (i, j) in edges:
        part1 = PSS[(i, j)] * (1 - qij[(i,j)] * qij_D[(i, j)]) * (1 - qij[(j,i)] * qij_D[(j, i)])
        part2 = PSI[(i, j)] * (1 - (1 - beta) * qij[(i, j)] * uij[(i,j)] * qij_D[(i,j)]) * (1 - gamma)
        part3 = PIS[(i, j)] * (1 - (1 - beta) * qij[(j, i)] * uij[(j,i)] * qij_D[(j,i)]) * (1 - gamma)
        part4 = PII[(i, j)] * (1 - gamma)**2
        PII_new[(i, j)] = part1 + part2 + part3 + part4

def _Hror_dSIS_ELE_base(gamma, 
                        beta, 
                        beta_D,
                        PI, 
                        PII,
                        node_neighbors_dict, 
                        tri_neighbors_dict):
    N = len(PI)
    edges = np.array(list(PII.keys()))

    PIS = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    PSI = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    PSS = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_P(PI, PII, PIS, PSI, PSS)

    PSII = Dict.empty(key_type=types.UniTuple(types.int64, 3), value_type=types.float64)
    update_PSII(PI, PSI, PII, tri_neighbors_dict, PSII)

    qi = Dict.empty(key_type=types.int64, value_type=types.float64)
    qi_D = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_qi_qiD(N, PI, PSI, PSII, beta, beta_D, node_neighbors_dict, tri_neighbors_dict, qi, qi_D)
    

    qij = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    qij_D = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    uij = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_qij_qijD_uij(edges, PI, PSI, PSII, beta, beta_D, qi, qi_D, tri_neighbors_dict, qij, qij_D, uij)

    PI_new = Dict.empty(key_type=types.int64, value_type=types.float64)
    update_PI(N, PI, gamma, qi, qi_D, PI_new)

    PII_new = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    update_PII(edges, PSS, PSI, PIS, PII, qij, qij_D, uij, beta, gamma, PII_new)
    
    return PI_new, PII_new


def Hror_SIS_ELE(beta,
                 beta_D,
                 gamma,
                 node_neighbors_dict,
                 tri_neighbors_dict,
                 tmax,
                 I0,
                 steady=True):
    node_neighbors_dict_nb, tri_neighbors_dict_nb = convert2nb_dict(node_neighbors_dict, tri_neighbors_dict)

    N = len(node_neighbors_dict)
    all_nodes = node_neighbors_dict.keys()

    PI = Dict.empty(key_type=types.int64, value_type=types.float64)
    for i in all_nodes:
        PI[i] = 1 if i in I0 else 0
    
    PII = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    for i in all_nodes:
        for j in node_neighbors_dict[i]:
            if i in I0 and j in I0:
                PII[(i,j)] = 1
            else:
                PII[(i,j)] = 0
    
    rho = np.zeros(tmax)
    rho[0] = sum(PI.values()) / N
    for t in range(1, tmax):
        PI, PII = _Hror_dSIS_ELE_base(gamma,
                                    beta,
                                    beta_D,
                                    PI,
                                    PII,
                                    node_neighbors_dict_nb,
                                    tri_neighbors_dict_nb)
        rho[t] = sum(PI.values()) / N
        if ifConverge(rho[:t], N):
            rho = rho[:t]
            break
    
    if steady:
        return beta_D, I0, beta, np.mean(rho[-100:])
    else:
        return beta_D, I0, beta, rho

if __name__ == "__main__":
    from load_data import import_random_ersc
    node_neighbors_dict, tri_neighbors_dict, avg_k1, avg_k2 = import_random_ersc(n=2000, k1=20, k2=6)
    gamma = 0.05

    lambda_Ds = np.array([2.5, 2.5, 0.8])
    lambdas = np.linspace(0.2, 2.2, 30)

    betas = (gamma/avg_k1) * lambdas
    beta_Ds = (gamma/avg_k2) * lambda_Ds

    I0_percent = np.array([0.4, 0.01, 0.01])
    N = len(node_neighbors_dict)
    I0s = [np.random.choice(range(N), size=int(N*I0p)) for I0p in I0_percent]

    import time
    t0 = time.time()
    Hror_SIS_ELE(betas[0], beta_Ds[0], gamma, node_neighbors_dict, tri_neighbors_dict, tmax=6000, I0=I0s[0])
    print("time = {:.3}s".format(time.time()-t0))