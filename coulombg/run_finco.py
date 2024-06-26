# -*- coding: utf-8 -*-
"""
A prototype script for running FINCO propagation on an initial grid of trajectories.
There are two options for propagation in the script, the first going over all the
"orders" from 0 to N, and the ssecond going only over one given "order" n. The results
are then saved to files, and the result objects are kept.

@author: Noam Ottolenghi
"""

#%% Setup

import logging

import numpy as np

from coulombg import V, S0, m, n_jobs, CoulombGTimeTrajectory
from finco import propagate, create_ics

logger = logging.getLogger('finco')
logger.setLevel(logging.DEBUG)

#%%Run FINCO
X, Y = np.meshgrid(np.linspace(1e-10, 15, 150), np.linspace(-15, 15, 300))
# qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
qs = (X+1j*Y)[(Y != 1) & (Y != -1)]

N=10
results = [None] * N
for n in range(N):
    ics = create_ics(qs, S0 = S0)
    t = CoulombGTimeTrajectory(n=n).init(ics)
    qs = qs[np.argsort(np.abs(t.b))]
    results[n] = propagate(ics, V = V, m = m,
                            time_traj = CoulombGTimeTrajectory(n=n), gamma_f=1,
                            dt = 1e-4, drecord=1,
                            n_jobs = n_jobs, blocksize=2**10,
                            trajs_path=f'results/coulombg_{n}.hdf')

#%%Run FINCO 2
n = 5
X, Y = np.meshgrid(np.linspace(1e-10, 5, 502), np.linspace(-3.5, 3, 502))
# qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
qs = (X+1j*Y)[(Y != 1) & (Y != -1)]

test = propagate(create_ics(qs, S0 = S0),
                 V = V, m = m,
                 time_traj = CoulombGTimeTrajectory(n = n), gamma_f=1,
                 dt = 1e-4, drecord=1,
                 n_jobs=n_jobs, trajs_path='results/test.hdf')
