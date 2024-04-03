# -*- coding: utf-8 -*-
"""
Example of analytic following after the caustics in free potential (harmonic
oscillator with omega=0).

Propagates a supergaussian :math:`~e^{-x^4}` in harmonic potential and plots the
location of the caustics, calculated analytically and shown be looking on the
map of xi_1.

In addition, the example plots the prefactor map at time t=0 with a bright
overlay representing the factor, with a pair of physical and unphysical points
marked on it. Then it plots the reconstruction from each point, to show how
nonphysical contributions differ from the Gaussian they should represent.

@author: Noam Ottolenghi
"""

#%% Setup

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from finco import propagate, create_ics
from finco.stokes import calc_factor2, find_caustics
from utils import tripcolor_complex
from ho_supergaussian import S0, V, m, omega, HOTimeTrajectory

plt.rc('font', size=14)

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

# System params
gamma_f = 1

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
    a = m*omega/2
    rhs = -a/6*((gamma_f*np.cos(omega*t)+1j*a*np.sin(omega*t))/
                (1j*gamma_f*np.sin(omega*t)+a*np.cos(omega*t)))
    return [1+rhs**0.5, 1-rhs**0.5]

#%% Run

X, Y = np.meshgrid(np.linspace(-2, 4, 121), np.linspace(-3, 3, 121))

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = S0)

result = propagate(ics, V = V, m = m, gamma_f = gamma_f,
                   time_traj = HOTimeTrajectory(T=2*np.pi), dt = 1e-3, drecord=1/100, n_jobs=6,
                   trajs_path=None)

_, ax = plt.subplots(nrows=1, ncols=4, num='caustics-ho-supergaussian', figsize=(16,4))
for i, step in enumerate(np.linspace(0, 75, 4)):
    caustics = locate_caustics_analytic(t=step/100*2*np.pi)
    xi_1 = result.get_caustics_map(step).xi_1

    plt.sca(ax[i])
    tripcolor_complex(np.real(ics.q0), np.imag(ics.q0), xi_1, absmin=0.2)
    ax[i].scatter(np.real(caustics), np.imag(caustics), s=10)
    ax[i].set_xlabel(r'$\Re q_0$', fontsize=16)
    ax[i].set_xlim(-2, 4)
    ax[i].set_ylabel(r'$\Im q_0$', fontsize=16)
    ax[i].set_ylim(-3, 3)
    ax[i].set_title(rf'$t={step/100*2:.2f}\pi / \omega$')

plt.tight_layout()
plt.savefig('caustics-exploration/caustics-ho-supergaussian.png')

#%% Stokes forbidden locations
caustics = find_caustics(locate_caustics_analytic(t=0),
                         V = V, m = m, S0 = S0, time_traj=HOTimeTrajectory(),
                         dt=1e-3, gamma_f=1, n_jobs=2)

proj = result.get_projection_map(0)
trajs = result.get_trajectories(0,1)

S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
S_F *= (calc_factor2(caustics.iloc[0], proj.q0, proj.xi, proj.sigma * (np.imag(proj.q0) > 0)) *
        calc_factor2(caustics.iloc[1], proj.q0, proj.xi, proj.sigma * (np.imag(proj.q0) < 0)))

plt.figure('pref_caustics')
tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)
plt.tricontourf(np.real(trajs.q0), np.imag(trajs.q0), S_F, alpha=0.2, cmap='hot')

# The indices below were found manually for the configuration above. Change at your own risk!
idx_good = 7693
idx_bad = 5138
plt.scatter(np.real(proj.q0.iloc[idx_good]), np.imag(proj.q0.iloc[idx_good]), c='y', s=10)
plt.scatter(np.real(proj.q0.iloc[idx_bad]), np.imag(proj.q0.iloc[idx_bad]), c='red', s=10)

#%% Stokes forbidden illustration

x = np.linspace(-1, 4, 201)
qm = (2/np.pi)**0.25*np.exp(-(x-1)**4)
res = result.get_results(0,1)

_, (good, bad) = plt.subplots(1, 2, num='stokes-illustartion', figsize=(12,5))

# Calculate the Gaussian parameters, extracting the analytical prefactor and the one from FINCO
qf_good, pf_good = np.real(proj.xi.iloc[idx_good]) / 2 / gamma_f, -np.imag(proj.xi.iloc[idx_good])
gf_good = ((2*gamma_f / np.pi) ** 0.25 *
                   np.exp(-gamma_f*(x-qf_good)**2 + 1j * pf_good * (x - qf_good)))
finco_good = (np.exp(1j*(res.S.iloc[idx_good] +
                         res.p.iloc[idx_good]*(x-res.q.iloc[idx_good]) +
                         0.5*res.S_20.iloc[idx_good]*(x-res.q.iloc[idx_good])**2)))

# Difference between the analytical Gaussian and the FINCO Gaussian
diff_good = np.trapz(np.conj(gf_good)*qm, x) - np.trapz(np.conj(gf_good)*finco_good, x)

good.plot(x, np.real(np.conj(gf_good)*qm), 'C0')
good.plot(x, np.imag(np.conj(gf_good)*qm), ':C0')
good.plot(x, np.real(np.conj(gf_good)*finco_good), 'C1')
good.plot(x, np.imag(np.conj(gf_good)*finco_good), ':C1')

good.set_xlim(-1, 4)
good.set_xlabel(r'$x$')
good.set_ylim(-0.4, 0.8)
good.set_title(rf'$\Delta={diff_good:.3g}$')
# good.legend(['QM', '$g_f$', 'FINCO Gaussian approximation'])

# Calculate the Gaussian parameters, extracting the analytical prefactor and the one from FINCO
qf_bad, pf_bad = np.real(proj.xi.iloc[idx_bad]) / 2 / gamma_f, -np.imag(proj.xi.iloc[idx_bad])
gf_bad = ((2*gamma_f / np.pi) ** 0.25 *
                   np.exp(-gamma_f*(x-qf_bad)**2 + 1j * pf_bad * (x - qf_bad)))
finco_bad = (np.exp(1j*(res.S.iloc[idx_bad] +
                        res.p.iloc[idx_bad]*(x-res.q.iloc[idx_bad]) +
                        0.5*res.S_20.iloc[idx_bad]*(x-res.q.iloc[idx_bad])**2)))

# Difference between the analytical Gaussian and the FINCO Gaussian
diff_bad = np.trapz(np.conj(gf_bad)*qm, x) - np.trapz(np.conj(gf_bad)*finco_bad, x)

bad.plot(x, np.real(np.conj(gf_bad)*qm), 'C0')
bad.plot(x, np.imag(np.conj(gf_bad)*qm), ':C0')
bad.plot(x, np.real(np.conj(gf_bad)*finco_bad), 'C1')
bad.plot(x, np.imag(np.conj(gf_bad)*finco_bad), ':C1')

bad.set_xlim(-1, 4)
bad.set_xlabel(r'$x$')
bad.set_ylim(-0.4, 0.8)
bad.set_title(rf'$\Delta={diff_bad:.3g}$')
# bad.legend(['QM', '$g_f$', 'FINCO Gaussian approximation'])

plt.tight_layout()
plt.savefig('caustics-exploration/stokes-illustartion.png')
