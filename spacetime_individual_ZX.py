'''
Calculate the optimal precision for the global spacetime-individual battery
with the uniform superposition of the causal orders of 
phase channels e^{-iθZ/2} and e^{iθX/2} 
preceded by bit-flip channel ρ → 0.5 * ρ+ 0.5 * X ρ X
'''

import cvxpy as cp
import scipy.linalg
import itertools
import matplotlib.pyplot as plt
from constants import get_H
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
import sympy
from functools import cache
import csv
import multiprocessing as mp
from tqdm import tqdm
import sys



from utils import *
from numeric_utils import *


import joblib
import psutil
from parallel_tqdm import ParallelTqdm
from joblib import Parallel, delayed, Memory

memory = Memory(location='./cachedir', verbose=0)

total_threads = joblib.cpu_count()
mosek_threads = 2
worker_threads = total_threads // mosek_threads

def pdf(x):
    return 1 / (2*np.pi)
    # mu = 1
    # sigma = 0.1
    # return 1 / (sigma * sympy.sqrt(2 * np.pi)) * sympy.exp(-0.5 * ((x - mu) / sigma) ** 2)

@memory.cache
def construct_J():
    theta = sympy.symbols('theta')

    Omega = sympy.Matrix([
        [1],
        [0],
        [0],
        [1]
    ])

    Z = sympy.Matrix([
        [1, 0],
        [0, -1]
    ])

    Y = sympy.Matrix([
        [0, -1j],
        [1j, 0]
    ])

    X = sympy.Matrix([
        [0, 1],
        [1, 0]
    ])

    U_theta = sympy.exp(-0.5j * theta * Z)

    Kraus1 = [
        0.5 * sympy.eye(2),
        0.5 * X
    ]
    
    result_matrix1 = sympy.zeros(4, 4)
    for K in Kraus1:
        result_matrix1 += sympy.kronecker_product(sympy.eye(2), K) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), K).H

    C_theta_1 = sympy.kronecker_product(sympy.eye(2), U_theta) @ result_matrix1 @ sympy.kronecker_product(sympy.eye(2), U_theta).H

    V_theta = sympy.exp(0.5j * theta * X)
    Kraus2 = [
        0.5 * sympy.eye(2),
        0.5 * X
    ]

    result_matrix2 = sympy.zeros(4, 4)
    for K in Kraus2:
        result_matrix2 += sympy.kronecker_product(sympy.eye(2), K) @ Omega @ Omega.H @ sympy.kronecker_product(sympy.eye(2), K).H

    C_theta_2 = sympy.kronecker_product(sympy.eye(2), V_theta) @ result_matrix2 @ sympy.kronecker_product(sympy.eye(2), V_theta).H

    C_theta = sympy.kronecker_product(C_theta_1, C_theta_2)

    integrated = sympy.lambdify(theta, sympy.exp(-1j * theta) * C_theta.T * pdf(theta), 'numpy')

    J = integrate(integrated, 0, 2 * np.pi)
    return np.array(J, dtype=complex)

@memory.cache
def construct_prob(p, dim = 2):
    avg_cost = 0
    conds = []

    p = [p, 1 - p]
    lambda_x_real = [[cp.Parameter() for _ in range(2)] for _ in range(2)]
    lambda_x_imag = [[cp.Parameter() for _ in range(2)] for _ in range(2)]
    E = cp.Parameter() 

    comb_dim = [(1, 2), (2, 2), (2, 2)]
    tot_dim = np.prod(comb_dim)
    T_sup = cp.Variable((tot_dim*2, tot_dim*2), hermitian=True)
    T_list = []
    T_list_perm = []

    mul_val = [[None for _ in range(2)] for _ in range(2)]

    N = len(comb_dim)

    for perm_cnt in range(2):
        T = cp.Variable((tot_dim, tot_dim), hermitian=True)
        partial_T = [np.ones((1, 1))] + [
            partial_comb(T, comb_dim, n)
            for n in range(1, N+1)
        ]

        T_list.append(T)

        if perm_cnt == 0:
            T_list_perm.append(T)
        else:
            T_list_perm.append(permute(T, comb_dim, [0, 3, 4, 1, 2, 5]))

        # comb constraints
        conds += [
            ptrace(partial_T[n], comb_dim[:n], -1) == cp.kron(partial_T[n-1], np.eye(comb_dim[n-1][0]))
            for n in range(1, N+1)
        ]
        conds += [
            partial_T[-1] >> 0
        ]

        # # construct J
        # J = construct_J()
        # if perm_cnt > 0:
        #     J = permute(J, [dim]*4, [2, 3, 0, 1])

    T_sup_probe = ptrace(T_sup, np.reshape(comb_dim, (-1,)).tolist() + [2], [-1])
    conds += [
        T_sup_probe == sum(
            p[perm_cnt] * T_list_perm[perm_cnt]
            for perm_cnt in range(2)
        )
    ]
    conds += [
        T_sup >> 0
    ]
    
    # objective
    J = construct_J()
    for perm_cnt in range(2):
        for i in range(dim):
            J_comp = np.kron(J, 
                np.kron(fock_dm(dim, i), fock_dm(2, perm_cnt))
            )
            mul_val[perm_cnt][i] = cp.trace(T_sup @ J_comp)
            avg_cost += 2 * lambda_x_real[perm_cnt][i] * cp.real(mul_val[perm_cnt][i]) - 2 * lambda_x_imag[perm_cnt][i] * cp.imag(mul_val[perm_cnt][i])

    # energy constraints
    T_sup_energy = sum(
        p[perm_cnt] * T_list[perm_cnt]
        for perm_cnt in range(2)
    )
    H = lambda d : get_H(d)
    dim_in = [comb_dim[i][0] for i in range(N)]
    dim_out = [comb_dim[i][1] for i in range(N)]
    axes_in = [2*j for j in range(N)]
    axes_out = [2*j+1 for j in range(N)]
    for j, T in enumerate(T_list):
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
                ptrace(H_out @ T, comb_dim, axes_out) << H_in.T + E * np.eye(H_in.shape[0])
            ]

    # minimize objective
    obj = cp.Minimize(2 - avg_cost)
    prob = cp.Problem(obj, conds)

    try:
        prob.get_problem_data(solver=cp.MOSEK)
    except Exception:
        pass

    return prob, lambda_x_real, lambda_x_imag, E, mul_val, T_sup, T_list

def calc_superposition(args, dim=2):
    p = args['p'][0]

    prob, lambda_x_real, lambda_x_imag, E, mul_val, T_sup, T_list = construct_prob(p, dim)


    for i in range(2):
        for j in range(2):
            lambda_x_real[i][j].value = np.real(args['lambda_x'][i][j])
            lambda_x_imag[i][j].value = np.imag(args['lambda_x'][i][j])
    E.value = args['E']

    try:
        prob.solve(warm_start=True, solver=cp.MOSEK, verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': mosek_threads})
    except:
        return {
            'val': np.inf,
            'energy': E.value,
            'p': [p, 1 - p],
            'T_sup': None
        }
    
    sol = {
        'val': prob.value,
        'energy': E.value,
        'p': [p, 1 - p],
        'T_sup': T_sup
    }
    
    return sol

def solve_accurate():

    E_val = sorted([0.0125, 0.025] + np.linspace(0, 0.5, 10).tolist())
    lambda_val = np.exp(1j * np.linspace(0, 2*np.pi, 10))
    p_val = [0.5]
    res = []
    args = []

    for E in E_val:
        args += [
            {
                'p': [p, 1 - p],
                'lambda_x': np.reshape(lambda_x, (2, 2)).tolist(),
                'E': E,
            }
            for lambda_x in itertools.product(lambda_val, repeat=4)
            for p in p_val
        ]

    res = ParallelTqdm(n_jobs=worker_threads)(
        [delayed(calc_superposition)(arg) 
        for arg in args]
        )

    data = []
    opt_val = {
        (E, p): min([sol for sol in res if sol['energy'] == E and sol['p'][0] == p], key=lambda x: x['val'])
        for E in E_val
        for p in p_val
    }

    data = []
    try:
        with open('spacetime2.csv', 'r') as f:
            reader = csv.reader(f)
            # Read header
            header = next(reader)
            # Read data rows
            for row in reader:
                data.append(
                    {
                        h: float(row[i])
                        for i, h in enumerate(header)
                    }
                )

        col_name = 'val'
        p_name = 'p'
        if p_name not in header:
            header.append(p_name)
        if col_name not in header:
            header.append(col_name)
        for row in data:
            E = row['E']
            p = row['p']
            if row.get(col_name, None) is not None:
                if opt_val[(E, p)]['val'] < row[col_name]:
                    row[col_name] = opt_val[(E, p)]['val']
                    row[p_name] = opt_val[(E, p)]['p'][0]
            else:
                row[col_name] = opt_val[(E, p)]['val']
                row[p_name] = opt_val[(E, p)]['p'][0]
    except:
        header = ['E', 'p', 'val']
        data = [
            {
                'E': E,
                'p': p,
                'val': opt_val[(E, p)]['val'],  
            }
            for E in E_val
            for p in p_val
        ]

    with open('spacetime2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow([row[h] for h in header])


if __name__ == '__main__':
    memory.clear()
    solve_accurate()
    memory.clear()