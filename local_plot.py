'''
plot the van Trees bound and the optimal average Holevo cost with a narrow prior distribution for the phase channel
'''

import matplotlib.pyplot as plt
import csv
import numpy as np

def read_fisher(file_path: str):
    E_vals = []
    vt_vals = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            E = float(row[0])
            vt = float(row[2])
            E_vals.append(E)
            vt_vals.append(vt)
    return E_vals, vt_vals


def read_narrow(file_path: str):
    E_vals = []
    vals = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            E = float(row[0])
            val = float(row[2])
            E_vals.append(E)
            vals.append(val)
    return E_vals, vals


def main():
    # Read data
    fisher_E, van_trees = read_fisher('Fisher_information.csv')
    narrow_E, narrow_vals = read_narrow('narrow_distribution.csv')


    plt.plot(fisher_E, van_trees, label='the van Trees bound (for optimal Fisher information)')

    plt.plot(narrow_E, narrow_vals, label=f'optimal average Holevo cost')

    plt.xlabel('E')
    plt.ylabel('average cost')
    #ax.set_ylim(0, 1)
    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()
    plt.savefig('local_plot.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()

