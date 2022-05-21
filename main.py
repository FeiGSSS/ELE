# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/05/21 16:24:29
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np

from MC import Hror_SIS_MC
from MMCA import Hror_SIS_MMCA
from ELE import Hror_SIS_ELE
from loadata import import_random_ersc, import_random_sfsc

from utils import parser_mc_results, parser_results

from tqdm import tqdm
import pickle as pkl

import time
from multiprocessing import Pool


def run(net, beta2):
    net_results = {"beta":beta1}
    node_nei, tri_nei, _, _ = net
    n = len(node_nei)
    I0 = np.random.choice(range(n), size=int(n*rho0), replace=False)

    print("ELE ...")
    t0 = time.time()
    pool = Pool(min(len(beta1), cpu_cores))
    results = []
    for beta in beta1:
        results.append(pool.apply_async(Hror_SIS_ELE, (beta, beta2, gamma, node_nei, tri_nei, tmax, I0)))
    pool.close()
    pool.join()
    rho = {}
    for (beta, v) in [r.get() for r in results]:
        rho[beta] = v
    net_results["ele"] = parser_results(rho)
    t1 = time.time()
    print("\nELE done, time = {:.1f}s".format(t1-t0))

    print("MMCA  ...")
    pool = Pool(min(len(beta1), cpu_cores))
    results = []
    for beta in beta1:
        results.append(pool.apply_async(Hror_SIS_MMCA, (beta, beta2, gamma, node_nei, tri_nei, tmax, I0)))
    pool.close()
    pool.join()
    rho = {}
    for (k, v) in [r.get() for r in results]:
        rho[k] = v
    net_results['mmca']  = parser_results(rho)
    print("MMCA Done!")

    print("Simulation ...")
    rhos = {}
    for beta in tqdm(beta1):
        rhos[beta] = Hror_SIS_MC(beta, beta2, gamma, node_nei, tri_nei, tmax, I0, cpu_cores, mc_num, steady=True)
    net_results["simulation"] = parser_mc_results(rhos, cut=True)
    print("Simulation done")

    return net_results

if __name__ == "__main__":
    cpu_cores = 50 # number of cpu cores to be used
    mc_num = 100   # number of independent simulation
    tmax = 80000   # the maximum steps of prapagation
    # rho0 = 0.4
    rho0 = 0.005

    gamma = 0.2
    er_beta2 = 0.1
    sf_beta2 = 0.14

    beta1 = np.logspace(-2.1, 0.0, 50)[:49]

    print("===========ER============")
    ER_net = import_random_ersc(n=2000, k1=12, k2=5)
    er_results = run(ER_net, er_beta2)
    with open("./results/er_results.pkl", "wb") as f:
        pkl.dump(er_results, f)

    print("===========SF============")
    SF_net = import_random_sfsc(n=8000, k1= 4, k2=3)
    sf_results = run(SF_net, sf_beta2)
    
    with open("./results/sf_results.pkl", "wb") as f:
        pkl.dump(sf_results, f)
    




