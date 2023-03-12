# Propagating the 1D Coulomb ground state with FINCO

Configuration and examples for propagation and reconstruction of the 1D Coulomb ground state with FINCO.

## Structure of this folder

This repository consists of the following:

1. `coulombg.py` - A configuration script containing the system's parameters, a class determining the trajectories in time and additional utility functions.
2. Several scripts for running propagation and analayzing it.
    - `run_finco.py` - a prototype script for running the propagation on a predefined set of initial conditions.
    - `run_finco_adaptive.py` - a script for running the propagation with adaptive sampling, and supporting command-line parameters to ease automation.
    - `caustic_times.py` - a script for running the process of finding the times a trajectory meets a caustic, and supporting command-line parameters to ease automation.
    - `analyze_results.py` - a prototype script for a general analysis of a single type of propagated trjaectories.
    - `coulombg_{1/1.5/2/3}cycle_recon.py` - scripts for reconstruction for the Coulomb ground state at 1/1.5/2/3 periods of the ground state.
3. Numerous scripts producing figures and examples of results.
    - `3cycles_orders_comparison.py` - produces figures comparing three contributing "orders" with the nonphysical trajectories already marked and manual filtering applied.
    - `order_2_exploration.py` - produces figures exploring "order" n=2 of the system.
    - `order_6_exploration.py` - produces figures exploring "order" n=6 of the system.
    - `plot_riemann_sheets.py` - produces figure exploring the Riemann sheets and poles in time for the real part of the momentum, to illustrate the behaviour in time.
    - `plot_trajectories_t_q.py` - produces figure comparing the trajectories in time for one initial condition, to show how trajectories behave.

For additional information refer to the documentation in the scripts.
