# -*- coding: utf-8 -*-
"""
Example of analytic following after the caustics in free potential (harmonic
oscillator with omega=0).

Propagates a supergaussian :math:`~e^{-x^4}` in free potential and plots the
location of the caustics, calculated analytically and shown be looking on the
map of xi_1

@author: Noam Ottolenghi
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics
from utils import tripcolor_complex
from ho_supergaussian import S0, m, HOTimeTrajectory

#%% Setup
plt.rc('font', size=14)

gamma_f = 1

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

# Small hack to create free potential (didn't want to make a diffrent folder for it...)
def free_V(q, _):
    return np.zeros_like(q)

V = [free_V, free_V, free_V]

def locate_caustics_analytic(t):
    """
    Calculates the caustics position analytically for given time

    Parameters
    ----------
    t : complex
        Time to calculate for.

    Returns
    -------
    caustics : list of length 2
        the 2 calculated caustics positions.
    """
    rhs = 1j * gamma_f / (6*(2*gamma_f/m*t-1j))
    return [1+rhs**0.5, 1-rhs**0.5]

#%% Run

X, Y = np.meshgrid(np.linspace(-2, 4, 61), np.linspace(-3, 3, 61))

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = S0)

result = propagate(ics, V = V, m = m, gamma_f = gamma_f,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)

_, ax = plt.subplots(nrows=1, ncols=3, num='caustics-free-supergaussian', figsize=(12,4))
for i, step in enumerate([0, 20, 100]):
    caustics = locate_caustics_analytic(t=step/100*10*np.pi)
    xi_1 = result.get_caustics_map(step).xi_1

    plt.sca(ax[i])
    tripcolor_complex(np.real(ics.q0), np.imag(ics.q0), xi_1, absmin=0.2)
    ax[i].scatter(np.real(caustics), np.imag(caustics), s=10)
    ax[i].set_xlabel(r'$\Re q_0$')
    ax[i].set_xlim(-2, 4)
    ax[i].set_ylabel(r'$\Im q_0$')
    ax[i].set_ylim(-3, 3)
    ax[i].set_title(rf'$t={int(step/100*10)}\pi / \omega$')

plt.tight_layout()
plt.savefig('caustics-exploration/caustics-free-supergaussian.png')
