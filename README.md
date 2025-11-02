# Optimal Quantum Metrology under Energy Constraints

This repository contains the code and data for the paper "Optimal Quantum Metrology under Energy Constraints" ([arXiv:2506.09436](https://arxiv.org/abs/2506.09436)). The project numerically investigates the ultimate precision limits of quantum metrology under energy constraints for different causal structures by semidefinite programming (SDP).

## Requirements

To run the code in this repository, you will need Python 3 and the following packages:

- **CVXPY:** For solving the semidefinite programs.
- **MOSEK:** A high-performance solver for convex optimization. (A license is required.)
- **NumPy & SciPy:** For numerical computations.
- **Matplotlib:** For generating plots.
- **SymPy:** For symbolic mathematics to calculate the task operator.
- **Joblib & tqdm:** For parallelization and progress bars.

## Code Description

- `constants.py`: Define the Hamiltonian used in the simulations.
- `utils.py` & `numeric_utils.py`: Contain various utility functions for quantum information and numerical calculations.

### Phase Estimation
- `phase.py`: Plot the optimal average Holevo cost for the phase estimation of different dimensions with arbitrary (or fixed) classical estimator. 

### Energy-efficiency Advantage of Causal Superposition
- `spacetime_individual_ZZ.py` & `spacetime_sharing_ZZ.py`: Solve the optimization problems for strategies employing a superposition of causal orders, with different battery configurations. The channels are $\mathrm{e}^{-\mathrm{i}\theta Z/2}$ and $\mathrm{e}^{\mathrm{i}\theta Z/2}$, both preceded by a bit-flip channel $\rho \to \frac{1}{2}\rho + \frac{1}{2}X\rho X$.
- `def_loc_ZZ.py` & `def_gl_ZZ.py`: Solve the optimization problems for strategies with a definite causal order, using local and global batteries, respectively. The channels are the same as above.
- `hierarchy_plot.py`: Generate the plot from the simulation data to show the energy-efficiency advantage of causal superposition.

### Separation between Battery Models
- `spacetime_individual_ZX.py` & `spacetime_sharing_ZX.py`: Solve the optimization problems for strategies employing a uniform superposition of causal orders, with different battery configurations. The channels are $\mathrm{e}^{-\mathrm{i}\theta Z/2}$ and $\mathrm{e}^{\mathrm{i}\theta X/2}$, both preceded by a bit-flip channel $\rho \to \frac{1}{2}\rho + \frac{1}{2}X\rho X$.
- `share_vs_ind_plot.py`: Plot the figure to show the separation between the spacetime-sharing and spacetime-individual batteries from the simulation data.

### Local Estimation
- `Fisher_information.py`: The script for calculating the Fisher information & the van-Trees bound. Plot the energy-constrained Fisher information with respect to energy.
- `narrow_distribution.py`: Determine the energy-constrained optimal average Holevo cost with a narrow Gaussian prior distribution.
- `local_plot.py`: Generate the plot from the simulation data to show the van-Trees bound and the energy-constrained average Holevo cost with respect to energy.

## How to Use

1.  **Set up the environment:** Make sure you have all the required packages installed.
2.  **Run the simulations:** Execute the main scripts to run the numerical optimizations. For example:

    ```bash
    python Fisher_information.py
    python narrow_distribution.py
    ```

    You can modify the parameters within each script to explore different scenarios.

3.  **Generate plots:** After running the simulations, the results will be saved to `.csv` files. You can then run the plotting scripts to visualize the data:

    ```bash
    python local_plot.py
    ```

## Note

Due to the exponential growth of the number of subproblems in the optimization with respect to the system dimension, it is common for computation time to be long when the system dimension > 2. 

