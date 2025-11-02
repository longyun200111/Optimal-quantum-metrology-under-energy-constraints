'''
plot the results for the example of phase channels e^{-iθZ/2} and e^{iθX/2} preceded by bit-flip channel ρ → 0.5 * ρ+ 0.5 * X ρ X
'''

import csv
import matplotlib.pyplot as plt
import numpy as np


with open('spacetime2.csv', 'r') as f:
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

plt.figure(figsize=(8, 4))

plt.plot(list(ind_gl.keys()), list(ind_gl.values()), label='causal superposition \n with global spacetime-individual battery', color='tomato')
plt.plot(list(sh_gl.keys()), list(sh_gl.values()), label='causal superposition \n with global spacetime-sharing battery', color='tomato', linestyle='--')



plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.xlabel('energy')
plt.ylabel('average cost')
plt.savefig('share_vs_ind.pdf', dpi=600)
plt.show()
