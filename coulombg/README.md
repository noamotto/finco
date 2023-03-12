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
    - 

In addition, a simplistic implementation of SPO is provided in `splitting_method.py`, which can be used for comparisons.

## Prerequisites

In order to run this library and its examples you'll need `python >= 3.8` as well as the following dependencies:
- `joblib >= 1.2.0` used for multiprocessing
- `matplotlib >= 3.6.0` used for plotting intermediate results
- `numpy >= 1.20.3`
- `scipy >= 1.9.1`
- `tqdm >= 4.64.1` 
- `pandas >= 1.5.0` used for dataset management. Should be intalled with `pytables >= 3.7.0` for result saving and retrieving.
- `typing >= 3.10.0.0`
- `scikit-image >= 0.19.3` used in legacy code treating Stokes phenomena
- `mayavi2 >= 4.8.1` used to produce 3D figures.

The library also contains a python extension that should be compiled via Cython. The compilation is done using `setuptools >= 65.4.0` and `cython >= 0.29.32`. Windows users will need to provide a toolchain for this compilation, either using MinGW, clang or MSVC. Using conda one can install a MinGW toolchain using 

> conda install -c conda-forge m2w64-toolchain

which works, but one should note that it is quite old.

## How to use this repository

In order to use this repository, simply clone it into your machine. After that, change directory into `finco/` and run
> python setup.py build_ext --inplace

That will build the python extension needed for the project.
You are then free to run any of the example scripts, according to their description.

## General pipeline when propagating with FINCO

In general, there are three things required to run a FINCO propagation:
1. The system's parameters, given as the initial state and the potential, as well as their 2 first spatial derivatives. Those are needed to sample the
    initial conditions and propagate the trajectories.
2. A set of initial positions to sample.
3. A class determining the trajectory in complex time for each initial condition.

Using these the propagation can be done directly by calling either `propagate` or `adaptive_sampling`. The propagation results can then be either saved to file or in memory.

The next step will usually be to deal with nonphysical contributions. That is currently done by following the procedure described in https://aip.scitation.org/doi/pdf/10.1063/1.5024467.
There are several tools for doing so in `finco.stokes`, which allow for finding the caustics in the propagated results, calculation of all the necessary quantities and calculating the factors that should be applied on each trajectory.

Finally, you can reconstruct the wavefunction using those factors by calling `reconstruct_psi` of the propagation results object, to get the reconstructed wavefunction at given points in space.

## Limitations and Future Work

As FINCO is still in development, there is still a lot of room for improvement.

The main limitation of the method currently is that it works well for one degree of freedom, and therefore the code here works only for one degree of freedom. We currently work on generalizing the method to variable number of degrees of freedom.

As for this codebase, it was mainly built so far for my needs as a student. As such, there are probably several issues and limitations concerning the code that can be addressed. You are more than welcome to open issues in this repository.
