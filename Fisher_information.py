'''
Calculate and plot the Fisher information (& the van Trees bound) for the phase channel
'''


import cvxpy as cp
from matplotlib.image import resample
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
from constants import get_H
import numpy as np
import scipy.sparse as sp
import sympy
from functools import cache
import csv
from tqdm import tqdm
import multiprocessing as mp

from joblib import Parallel, delayed, Memory

from utils import *
from numeric_utils import *

mem = Memory('./cachedir', verbose=0)

def pdf(x):
    return 1 / (2*np.pi)
    # mu = 1
    # sigma = 0.1
    # return 1 / (sigma * sympy.sqrt(2 * np.pi)) * sympy.exp(-0.5 * ((x - mu) / sigma) ** 2)

@mem.cache
def construct_C(theta_val=np.pi, is_symbol=False):
    theta = sympy.symbols('theta', real=True)

    Omega = sympy.Matrix([
        [1],
        [0],
        [0],
        [1]
    ])

    U_theta = sympy.Matrix([
        [1, 0],
        [0, sympy.exp(sympy.I * theta)]
    ])

    C_theta = sympy.kronecker_product(sympy.eye(2), U_theta) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), U_theta).H

    dC_theta = sympy.diff(C_theta, theta)

    if is_symbol:
        return C_theta, dC_theta
    else:
        C_theta = C_theta.subs(theta, theta_val)
        dC_theta = dC_theta.subs(theta, theta_val)

        return np.array(C_theta, dtype=complex), np.array(dC_theta, dtype=complex)

@mem.cache
def construct_prob(sign):
    avg_cost = 0
    conds = []

    p = cp.Parameter()
    E = cp.Parameter() 

    comb_dim = [(1, 2), (2, 2)]

    N = len(comb_dim)

    T = cp.Variable((np.prod(comb_dim), np.prod(comb_dim)), hermitian=True)
    partial_T = [np.ones((1, 1))] + [
            partial_comb(T, comb_dim, n)
            for n in range(1, N+1)
        ]

    # comb constraints
    conds += [
        ptrace(partial_T[n], comb_dim[:n], -1) == cp.kron(partial_T[n-1], np.eye(comb_dim[n-1][0]))
        for n in range(1, N+1)
    ]
    conds += [
        T >> 0
    ]
        
    # objective
    C, dC = construct_C()
    for i in range(2):
        C_comp = np.kron(C, 
            fock_dm(2, i)
        )
        dC_comp = np.kron(dC, 
            fock_dm(2, i)
        )
        if i == 0:
            conds += [
                cp.trace(C_comp @ T) == p
            ]
            avg_cost = sign * cp.real(cp.trace(dC_comp @ T))

    # energy constraints
    T_sup_energy = T
    H = lambda d : get_H(d)
    dim_in = [comb_dim[i][0] for i in range(N)]
    dim_out = [comb_dim[i][1] for i in range(N)]
    axes_in = [2*j for j in range(N)]
    axes_out = [2*j+1 for j in range(N)]
    for i in range(N):
        H_in = sum(
            extend(H(dim_in[j]), dim_in, j)
            for j in range(i + 1)
        )
        H_out = sum(
            extend(H(dim_out[j]), comb_dim, axes_out[j])
            for j in range(i + 1)
        )
        conds += [
            ptrace(H_out @ T_sup_energy, comb_dim, axes_out) << H_in.T + E * np.eye(H_in.shape[0])
        ]

    # minimize objective
    obj = cp.Maximize(avg_cost)
    prob = cp.Problem(obj, conds)

    FI = (cp.real(cp.trace(dC_comp @ T)) ** 2) * (1 / p + 1 / (1-p))

    return prob, p, E, FI, T
    

def calc(args):
    max_sol = None
    for sign in [-1, 1]:
        prob, p, E, FI, T = construct_prob(sign)

        p.value = args['p']
        E.value = args['E']
        
        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
        except cp.error.SolverError:
            continue

        if prob.status != cp.OPTIMAL:
            continue

        sol = {
            'val': FI.value,
            'energy': E.value,
            'p': p.value,
            'T': T.value
        }

        if max_sol is None or sol['val'] > max_sol['val']:
            max_sol = sol

    

    #print(f'energy: {E.value}')

    #sol['Tx'] = [Tx[i].value for i in range(dim)]
    #sol['rho'] = rho.value
    
    return max_sol

def calc_van_trees(T, sigma=1, mu=np.pi):
    theta = sympy.symbols('theta', real=True)
    gauss_p = 1 / (sigma * np.sqrt(2 * np.pi)) * sympy.exp(-0.5 * ((theta - mu) / sigma) ** 2)
    gauss_p_diff = sympy.diff(gauss_p, theta)

    I_Q = integrate(
        sympy.lambdify(theta, (gauss_p_diff ** 2) / gauss_p, 'numpy'),
        -np.inf, np.inf
    )

    C_theta, dC_theta = construct_C(theta_val=None, is_symbol=True)
    p = sympy.trace(sympy.kronecker_product(C_theta, sympy.Matrix(fock_dm(2,0))) @ T)
    p_diff = sympy.trace(sympy.kronecker_product(dC_theta, sympy.Matrix(fock_dm(2,0))) @ T)
    I_p = (p_diff ** 2) * (1 / p + 1 / (1-p))

    I_P_int = integrate(
        sympy.lambdify(theta, I_p * gauss_p, 'numpy'),
        -np.inf, np.inf
    )

    return np.real(1 / (I_Q + I_P_int))

def solve_accurate():
    p_val = np.linspace(1e-5, 1 - 1e-5, 15).tolist()
    E_val = sorted(np.linspace(1e-5, 1, 10).tolist() + np.linspace(1e-5, 0.2, 10).tolist())
    
    res_accurate = []

    args = [
        {
            'p': p,
            'E': E,
        }
        for p in p_val
        for E in E_val
        if E >= 1 - p
    ]
        
    res_accurate = Parallel(n_jobs=-1)(
        delayed(calc)(arg) 
        for arg in tqdm(args)
    )
    
    opt_val = {
        E: max(val_list, key=lambda x: x['val'])
        for E in E_val
        if (val_list := [sol for sol in res_accurate if sol is not None and sol['energy'] == E])
    }
    with open('Fisher_information.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['E', 'val', 'van_trees'])
        sigma = 0.5
        for E in E_val:
            if opt_val.get(E, None) is None:
                continue
            cr_bound = 1 / opt_val[E]['val']
            van_trees = calc_van_trees(opt_val[E]['T'], sigma)
            writer.writerow([E, cr_bound, van_trees])

    plt.plot(list(opt_val.keys()), [1 / opt_val[E]['val'] for E in opt_val.keys()])
    plt.xlabel('E')
    plt.ylabel(r'$\mathrm{Var}(\hat{\theta}-\theta) * \nu_{\text{rep}}$')
    plt.ylim(0, 20)
    #plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('Fisher_information.pdf', dpi=600)
    plt.show()
    
if __name__ == '__main__':
    solve_accurate()