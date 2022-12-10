# FINCO
An implementation of the FINCO method in 1D and some example useages

## Structure of this repository
---
This repository consists of two main parts:

1. The main library code, in the `finco/` folder
2. Usage examples of three different systems:
    - Harmonic oscillator reconstruction over time, in `harmonic-oscillator/`
    - Quartic oscillator reconstruction at a specific time, in `quartic/`
    - Coulomb ground state propagation and reconstruction, in `coulombg/`

In addition, a simplistic implementation of SPO is provided in `splitting_method.py`, which can be used for comparisons.

## Prerequisites
---

In order to run this library and its examples you'll need `python>=3.8` as well as the following dependencies:
- `joblib >= 1.2.0` used for multiprocessing
- `matplotlib >= 3.6.0` used for plotting intermediate results
- `numpy >= 1.20.3`
- `scipy >= 1.9.1`
- `tqdm >= 4.64.1` 
- `pandas >= 1.5.0` used for dataset management. Should be intalled with `pytables >= 3.7.0` for result saving and retrieving.
- `typing >= 3.10.0.0`
- `scikit-image >= 0.19.3` used in legacy code treating Stokes phenomena

The library also contains a python extension that should be compiled via Cython. The compilation is done using `setuptools >= 65.4.0` and `cython >= 0.29.32`. Windows users will need to provide a toolchain for this compilation, either using MinGW, clang or MSVC. Using conda one can install a MinGW toolchain using 

> conda install -c conda-forge m2w64-toolchain

which works, but one should note that it is quite old.

## How to use this repository
---
In order to use this repository, simply clone it into your machine. After that, change directory into `finco/` and run
> python setup.py build_ext --inplace

That will build the python extension needed for the project.
You are then free to run any of the example scripts, according to their description.