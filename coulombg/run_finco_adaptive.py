# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

from coulombg import halfcycle, V, S0, m, n_jobs, CoulombGTimeTrajectory

import logging
import argparse
import os

import numpy as np

from finco import adaptive_sampling


def main():
    logging.getLogger('finco').setLevel(logging.INFO)    
    
    parser = argparse.ArgumentParser(description="Runs adaptive sampling for Coulomb ground state system")
    parser.add_argument('n', type=int, action='store', help="Order of circling poles to compute for")
    parser.add_argument('-t', action='store', type=np.float64, default=3,
                        help="Final time for propagation, in cycles")
    parser.add_argument('-o', '--outdir', action='store', type=str, default='.',
                        help="Results directory")
    args = parser.parse_args()
    
    n_iters = 7
    n_steps = 1
    sub_tol = (2e-1, 1e3)
    conv_E = 1e-2
    conv_N = 5
    X, Y = np.meshgrid(np.linspace(1e-10, 15, 150), np.linspace(-15, 15, 300))
    
    adaptive_sampling(qs = (X+1j*Y).flatten(), S0 = S0,
                      n_iters = n_iters, sub_tol = sub_tol,
                      conv_E = conv_E, conv_N = conv_N, plot_steps=True,
                      filter_func = lambda q: np.abs(q) > 0.01,
                      V = V, m = m, gamma_f = 1,
                      time_traj = CoulombGTimeTrajectory(n=args.n, t=halfcycle*2*args.t),
                      dt = 1e-4, drecord=1 / n_steps, n_jobs=n_jobs,
                      trajs_path=os.path.join(args.outdir, f'coulombg_{args.n}.hdf'))
    
if __name__ == "__main__":
    main()
