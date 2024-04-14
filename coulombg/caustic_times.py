# -*- coding: utf-8 -*-
"""
Runs caustic times finder for a set of trajectories in Coulomb ground state, 
with command line arguments to allow automation.

The script is built such that it could be executed as a part of automation script,
allowing one to set the results file to use as an initial state of the system while
running the caustic time finding algorithm.

@author: Noam Ottolenghi
"""

#%% Setup
import logging
import argparse

import numpy as np

from finco import load_results
from finco.stokes import caustic_times
from coulombg import V, m, n_jobs
from coulombg import coulombg_caustic_times_dir, coulombg_caustic_times_dist

def main():
    logging.getLogger('finco').setLevel(logging.INFO)    
    
    parser = argparse.ArgumentParser(description="Runs caustic time finder for CoulombG results file")
    parser.add_argument('file', type=str, action='store', help="Result file to load")
    args = parser.parse_args()    
    

    result = load_results(args.file)    
    caustic_times(result, coulombg_caustic_times_dir, coulombg_caustic_times_dist, n_iters = 1500,
                  skip = 150, plot_steps=False, x=np.linspace(1e-10,15,500),
                  V = V, m = m, gamma_f=1, dt=1, 
                  n_jobs=n_jobs, blocksize=2**12,
                  verbose=False)

if __name__ == "__main__":
    main()