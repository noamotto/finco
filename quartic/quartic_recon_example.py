# -*- coding: utf-8 -*-
"""
Example of propagation, treatment and reconstruction of wavefunction in quatric
potential.

Should be compared against the result of SPO propagation, by running SPOquartic.py
with the same initial conditions and time.

@author: Noam Ottolenghi
"""
#%% Setup
import logging

import numpy as np
import matplotlib.pyplot as plt

from quartic import S0, V, m, QuarticTimeTrajectory, eliminate_stokes
from finco import propagate, create_ics, load_results

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)

gamma_f = 1
T = 0.72
x = np.linspace(-5, 5,1000)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 201), np.linspace(-2.5, 2.5, 201))
qs = (X+1j*Y).flatten()
ics = create_ics(qs, S0 = S0)

#%% Propagate
result = propagate(ics, V = V, m = m, gamma_f=gamma_f,
                   time_traj = QuarticTimeTrajectory(T = T), dt = 1e-3, drecord=1,
                   blocksize=2**9, n_jobs=3, verbose=True)

#%% Treat stokes and plot
trajs = load_results('trajs.hdf', gamma_f=1).get_trajectories(1)

S_F1 = eliminate_stokes(result)
plt.figure('factor')
plt.tripcolor(np.real(trajs.q0), np.imag(trajs.q0), S_F1, shading='gouraud')
plt.xlabel('$\Re q_0$')
plt.ylabel('$\Im q_0$')
plt.title('Berry factor map')
plt.colorbar()

x = np.linspace(-5, 5,1000)
plt.figure('recon')
plt.plot(x, np.abs(result.reconstruct_psi(x, 1, S_F1, n_jobs=3)))
plt.xlabel('$x$')
plt.ylabel('$|\Psi|$')
plt.title('Reconstruction')
