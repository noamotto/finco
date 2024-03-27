# -*- coding: utf-8 -*-
"""
Example of producing caustic times

The code runs a simple propagation of initial conditions in quartic potential
and finds their caustic times, plotting the progress every 10 iterations.

@author: Noam Ottolenghi
"""

#%% Setup
import logging

import numpy as np

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import propagate, create_ics
from finco.stokes import caustic_times

def quartic_caustic_times_dist(q0, p0, t0, est):
    return np.full_like(q0, 1e-1)

def quartic_caustic_times_dir(q0 ,p0 ,t0, est):
    return np.full_like(q0, 1)

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)

gamma_f = 1
T = 0.72
x = np.linspace(-5, 5,1000)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 201), np.linspace(-2.5, 2.5, 201))
qs = (X+1j*Y).flatten()
ics = create_ics(qs, S0 = S0)

#%% Propagate and find caustic times
result = propagate(ics, V = V, m = m, gamma_f=gamma_f,
                   time_traj = QuarticTimeTrajectory(T = T), dt = 1e-3, drecord=1,
                   blocksize=2**9, n_jobs=3, verbose=True)

ts = caustic_times(result, quartic_caustic_times_dir, quartic_caustic_times_dist, n_iters = 130,
                   skip = 13, x = x, plot_steps=True,
                   V = V, m = m, gamma_f=1, dt=1, drecord=1,
                   n_jobs=3, blocksize=2**15,
                   verbose=False)
