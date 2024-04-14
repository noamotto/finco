# -*- coding: utf-8 -*-
"""
Adaptive sampling example

This code produces a "before" and "after" images of an adaptive sampling of
initial conditions.

The system taken is gaussian in quartic potential, and the images taken are the
prefactor complex-color images of the first and last steps of the subsampling.

@author: Noam Ottolenghi
"""

#%% Setup

import logging

import numpy as np
import matplotlib.pyplot as plt

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import load_results, adaptive_sampling
from utils import tripcolor_complex

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)

n_iters = 15
n_steps = 1
sub_tol = (1.5e-1,1e2)

#%% Run adaptive sampling
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 21), np.linspace(-2.5, 2.5, 21))
result, mesh = adaptive_sampling(qs = (X+1j*Y).flatten(), S0 = S0,
                                 n_iters = n_iters, sub_tol = sub_tol, plot_steps=True,
                                 V = V, m = m, gamma_f = 1,
                                 time_traj = QuarticTimeTrajectory(), dt = 1e-3,
                                 drecord=1 / n_steps, n_jobs=3)

#%% Prepare figure
trajs0 = load_results('trajs.hdf.steps/step_0.hdf', gamma_f=1).get_trajectories(1)
trajs = result.get_trajectories(1)

_, (n0, n) = plt.subplots(1, 2, num='adaptive-sampling-example', figsize=(10, 5))
plt.sca(n0)
tripcolor_complex(np.real(trajs0.q0), np.imag(trajs0.q0), trajs0.pref, absmax=1e7)
n0.set_xlim(-2.5, 2.5)
n0.set_xlabel(r'$\Re q_0$')
n0.set_ylim(-2.5, 2.5)
n0.set_ylabel(r'$\Im q_0$')

plt.sca(n)
tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)
n.set_xlim(-2.5, 2.5)
n.set_xlabel(r'$\Re q_0$')
n.set_ylim(-2.5, 2.5)
n.set_ylabel(r'$\Im q_0$')

#%% Finalize and save
plt.tight_layout()
# plt.savefig('adaptive-sampling-example')
