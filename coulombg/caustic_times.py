# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

from coulombg import V, m, n_jobs
from coulombg import coulombg_caustic_times_dir, coulombg_caustic_times_dist

import logging
import argparse

from finco import load_results

from finco.stokes import caustic_times

def main():
    logging.getLogger('finco').setLevel(logging.INFO)    
    
    parser = argparse.ArgumentParser(description="Runs caustic time finder for CoulombG results file")
    parser.add_argument('file', type=str, action='store', help="Result file to load")
    args = parser.parse_args()    
    

    result = load_results(args.file)    
    caustic_times(result, coulombg_caustic_times_dir, coulombg_caustic_times_dist, n_iters = 1500,
                  skip = 150, plot_steps=False,
                  V = V, m = m, gamma_f=1, dt=1, 
                  n_jobs=n_jobs, blocksize=2**13,
                  verbose=False)

if __name__ == "__main__":
    main()