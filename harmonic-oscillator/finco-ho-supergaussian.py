# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from finco import propagate, create_ics, TimeTrajectory
from utils import tripcolor_complex

#%% Setup

import os

plt.rc('font', size=14)

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

# System params
m = 1
omega = 1
gamma_f = 1

def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**4)
    
def S0_1(q):
    return 4j *(q - 1)**3

def S0_2(q):
    return 12j *(q - 1)**2

def V_0(q):
    return 0.5 * m * omega**2 * q ** 2
    
def V_1(q):
    return m * omega**2 * q

def V_2(q):
    return np.full_like(q,m * omega**2)

class HOTimeTrajectory(TimeTrajectory):
    def __init__(self, T = np.pi*2):
        self.T = T
        
    def init(self, ics):
        self.q0 = ics.q
        
    def t_0(self, tau):
        return np.full_like(self.q0, self.T*tau)
    
    def t_1(self, tau):
        return np.full_like(self.q0, self.T)
    
    def get_discontinuity_times(self):
        return []

def locate_caustics_analytic(t):
    a = m*omega/2
    rhs = -a/6*((gamma_f*np.cos(omega*t)+1j*a*np.sin(omega*t))/
                (1j*gamma_f*np.sin(omega*t)+a*np.cos(omega*t)))
    return [1+rhs**0.5, 1-rhs**0.5]

#%% Run

X, Y = np.meshgrid(np.linspace(-2, 4, 121), np.linspace(-3, 3, 121))

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2], gamma_f=1)

result = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f = gamma_f,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)

_, ax = plt.subplots(nrows=1, ncols=4, num='caustics-ho-supergaussian', figsize=(16,4))
for i, step in enumerate(np.linspace(0, 75, 4)):
    caustics = locate_caustics_analytic(t=step/100*2*np.pi)
    xi_1 = result.get_caustics_map(step).xi_1
    
    plt.sca(ax[i]), tripcolor_complex(np.real(ics.q0), np.imag(ics.q0), xi_1, absmin=0.2)
    ax[i].scatter(np.real(caustics), np.imag(caustics), s=10)
    ax[i].set_xlabel(r'$\Re q_0$', fontsize=16)
    ax[i].set_xlim(-2, 4)
    ax[i].set_ylabel(r'$\Im q_0$', fontsize=16)
    ax[i].set_ylim(-3, 3)
    ax[i].set_title(r'$t={:.2f}\pi / \omega$'.format(step/100*2))

plt.tight_layout()
plt.savefig('caustics-exploration/caustics-ho-supergaussian.png')

#%% Stokes forbidden locations

from finco.stokes import calc_factor2, find_caustics
caustics = find_caustics(locate_caustics_analytic(t=0), V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2],
                         time_traj=HOTimeTrajectory(), dt=1e-3, gamma_f=1,
                         n_jobs=2)

proj = result.get_projection_map(0)
trajs = result.get_trajectories(0,1)

S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
S_F *= (calc_factor2(caustics.iloc[0], proj.q0, proj.xi, proj.sigma * (np.imag(proj.q0) > 0)) * 
        calc_factor2(caustics.iloc[1], proj.q0, proj.xi, proj.sigma * (np.imag(proj.q0) < 0)))

plt.figure('pref_caustics')
tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)
plt.tricontourf(np.real(trajs.q0), np.imag(trajs.q0), S_F, alpha=0.2, cmap='hot')

# The indices below were found empirically for the configuration above. Change at your own risk!
idx_good = 7693
idx_bad = 5138
plt.scatter(np.real(proj.q0.iloc[idx_good]), np.imag(proj.q0.iloc[idx_good]), c='y', s=10)
plt.scatter(np.real(proj.q0.iloc[idx_bad]), np.imag(proj.q0.iloc[idx_bad]), c='red', s=10)

#%% Stokes forbidden illustration

x = np.linspace(-1, 4, 201)
qm = (2/np.pi)**0.25*np.exp(-(x-1)**4)
res = result.get_results(0,1)

_, (good, bad) = plt.subplots(1, 2, num='stokes-illustartion', figsize=(12,5))
qf_good, pf_good = np.real(proj.xi.iloc[idx_good]) / 2 / gamma_f, -np.imag(proj.xi.iloc[idx_good])
gf_good = ((2*gamma_f / np.pi) ** 0.25 * 
                   np.exp(-gamma_f*(x-qf_good)**2 + 1j * pf_good * (x - qf_good)))
finco_good = (np.exp(1j*(res.S.iloc[idx_good] + 
                         res.p.iloc[idx_good]*(x-res.q.iloc[idx_good]) + 
                         0.5*res.S_2.iloc[idx_good]*(x-res.q.iloc[idx_good])**2)))

good.plot(x, np.real(np.conj(gf_good)*qm), c=plt.cm.tab10(0))
good.plot(x, np.imag(np.conj(gf_good)*qm), ':', c=plt.cm.tab10(0))
good.plot(x, np.real(np.conj(gf_good)*finco_good), c=plt.cm.tab10(1))
good.plot(x, np.imag(np.conj(gf_good)*finco_good), ':', c=plt.cm.tab10(1))

good.set_xlim(-1, 4)
good.set_xlabel(r'$x$')
good.set_ylim(-0.4, 0.8)
good.set_title(rf'$\Delta={np.trapz(np.conj(gf_good)*qm, x) - np.trapz(np.conj(gf_good)*finco_good, x):.3g}$')
# good.legend(['QM', '$g_f$', 'FINCO Gaussian approximation'])

qf_bad, pf_bad = np.real(proj.xi.iloc[idx_bad]) / 2 / gamma_f, -np.imag(proj.xi.iloc[idx_bad])
gf_bad = ((2*gamma_f / np.pi) ** 0.25 * 
                   np.exp(-gamma_f*(x-qf_bad)**2 + 1j * pf_bad * (x - qf_bad)))
finco_bad = (np.exp(1j*(res.S.iloc[idx_bad] + 
                        res.p.iloc[idx_bad]*(x-res.q.iloc[idx_bad]) + 
                        0.5*res.S_2.iloc[idx_bad]*(x-res.q.iloc[idx_bad])**2)))

bad.plot(x, np.real(np.conj(gf_bad)*qm), c=plt.cm.tab10(0))
bad.plot(x, np.imag(np.conj(gf_bad)*qm), ':', c=plt.cm.tab10(0))
bad.plot(x, np.real(np.conj(gf_bad)*finco_bad), c=plt.cm.tab10(1))
bad.plot(x, np.imag(np.conj(gf_bad)*finco_bad), ':', c=plt.cm.tab10(1))

bad.set_xlim(-1, 4)
bad.set_xlabel(r'$x$')
bad.set_ylim(-0.4, 0.8)
bad.set_title(rf'$\Delta={np.trapz(np.conj(gf_bad)*qm, x) - np.trapz(np.conj(gf_bad)*finco_bad, x):.3g}$')
# bad.legend(['QM', '$g_f$', 'FINCO Gaussian approximation'])

plt.tight_layout()
plt.savefig('caustics-exploration/stokes-illustartion.png')
