'''
plot the results for the example of phase channels e^{-iθZ/2} and e^{iθZ/2} preceded by bit-flip channel ρ → 0.5 * ρ+ 0.5 * X ρ X
'''

import csv
import matplotlib.pyplot as plt
import numpy as np


def_global = {}
def_local = {}

with open('def_gl.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)
    for row in data:
        E = float(row[0])
        val = float(row[1])
        def_global[E] = val

with open('def_loc.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)
    for row in data:
        E = float(row[0])
        val = float(row[1])
        def_local[E] = val

with open('spacetime1.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    E_vals = sorted({float(row['E']) for row in data})
    ind_gl = {
        E : min(
            float(row['val']) for row in data if float(row['E']) == E
        )
        for E in E_vals
    }
    sh_gl = {
        E : min(
            float(row['val_shared']) for row in data if float(row['E']) == E
        )
        for E in E_vals
    }
    def_cal_anc = {
        E : min(
            float(row['val']) for row in data 
            if float(row['E']) == E and (float(row['p']) == 0 or float(row['p']) == 1)
        )
        for E in E_vals
    }

plt.figure(figsize=(8, 4))

plt.plot(list(def_local.keys()), list(def_local.values()), label='definite causal order with local battery \n (w/o access to the causality qubit)', color='black', linestyle='--')
plt.plot(list(def_global.keys()), list(def_global.values()), label='definite causal order with global battery \n (w/o access to the causality qubit)', color='black')
plt.plot(list(def_cal_anc.keys()), list(def_cal_anc.values()), label='definite causal order with global battery \n (w/ access to the causality qubit)', color='royalblue')
plt.plot(list(ind_gl.keys()), list(ind_gl.values()), label='causal superposition \n with global spacetime-individual battery', color='tomato')


plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.xlabel('energy')
plt.ylabel('average cost')
plt.savefig('hierarchy.pdf', dpi=600)
plt.show()
