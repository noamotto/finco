# -*- coding: utf-8 -*-
"""
Basic FINCO propagation and reconstruction example.

Propagates a Gaussian in harmonic potential for one period,
and reconstructs and plots the result.

@author: Noam Ottolenghi
"""

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics
from ho_supergaussian import V, m, HOTimeTrajectory

#%% Setup

# System parameters
omega = 1

def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**2)

def S0_1(q):
    return 2j * (q - 1)

def S0_2(q):
    return np.full_like(q,2j)

S0 = [S0_0, S0_1, S0_2]

#%% Run
X, Y = np.meshgrid(np.linspace(-2, 4, 41), np.linspace(-3, 3, 61))

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = S0, gamma_f=1)

result = propagate(ics, V = V, m = m, gamma_f = 1,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)

x = np.arange(-10, 10, 1e-2)
y = result.reconstruct_psi(x, 100)

plt.plot(x, np.abs(y))
