# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finco import TimeTrajectory, load_results, adaptive_sampling
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2
from utils import tripcolor_complex

plt.rc('font', size=14)

# System params
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))
    
def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))

def S0_2(q):
    return np.full_like(q, 2j*gamma0)

def V_0(q):
    return a*q**2 + b*q**4
    
def V_1(q):
    return 2*a*q + 4*b*q**3

def V_2(q):
    return 2*a + 12*b*q**2

class QuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics):        
        self.t = np.full_like(ics.q, 0.72)
        
    def t_0(self, tau):
        return self.t * tau
        
    def t_1(self, tau):
        return self.t

class HalfQuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics): 
        self.t0 = ics.t.to_numpy()
        self.t = np.full_like(ics.q, 0.72/2) + self.t0
        
    def t_0(self, tau):
        return (self.t - self.t0) * tau + self.t0
        
    def t_1(self, tau):
        return self.t - self.t0
    
def eliminate_stokes(result):
    # Load projection map, map to a grid, and calculate F
    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)

    # plt.figure()
    # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

    blobs = separate_to_blobs(deriv, quantile=1e-2)
    qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]
        
    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2], 
                             time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3)
    # caustics = caustics[np.real(caustics.q) > 0]
    
    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (i, caustic) in caustics.iterrows():
        # idx = np.argmin(np.abs(proj.q0-caustic.q))
        # caustic.q = proj.q0.iat[idx]
        # caustic.xi = proj.xi.iat[idx]
        S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)
    
    return S_F


#%%
import logging

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)
n_iters = 7
n_steps = 1
sub_tol = (2e-1,1e3)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 21), np.linspace(-2.5, 2.5, 21))
result, mesh = adaptive_sampling(qs = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2],
                                 n_iters = n_iters, sub_tol = sub_tol, plot_steps=False,
                                 V = [V_0, V_1, V_2], m = m, gamma_f = 1,
                                 time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1 / n_steps, 
                                 n_jobs=3)

#%%
trajs0 = load_results('trajs.hdf.steps/step_0.hdf', gamma_f=1).get_trajectories(1)
trajs = result.get_trajectories(1)

_, (n0, n) = plt.subplots(1, 2, num='adaptive-sampling-example', figsize=(10, 5))
plt.sca(n0), tripcolor_complex(np.real(trajs0.q0), np.imag(trajs0.q0), trajs0.pref, absmax=1e7)
n0.set_xlim(-2.5, 2.5)
n0.set_xlabel(r'$\Re q_0$')
n0.set_ylim(-2.5, 2.5)
n0.set_ylabel(r'$\Im q_0$')

plt.sca(n), tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)
n.set_xlim(-2.5, 2.5)
n.set_xlabel(r'$\Re q_0$')
n.set_ylim(-2.5, 2.5)
n.set_ylabel(r'$\Im q_0$')

#%%
plt.tight_layout()
# plt.savefig('adaptive-sampling-example')
