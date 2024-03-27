# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

from finco import TimeTrajectory
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2


# System params
hbar = 1
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2 -
                (chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))*hbar

def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))*hbar

def S0_2(q):
    return np.full_like(q, 2j*gamma0)*hbar

S0 = [S0_0, S0_1, S0_2]

def V_0(q, t):
    return a*q**2 + b*q**4

def V_1(q, t):
    return 2*a*q + 4*b*q**3

def V_2(q, t):
    return 2*a + 12*b*q**2

V = [V_0, V_1, V_2]

class QuarticTimeTrajectory(TimeTrajectory):
    def __init__(self, T = 0.72):
        self.T = T

    def init(self, ics):
        self.t = np.full_like(ics.q, self.T)

    def t_0(self, tau):
        return self.t * tau

    def t_1(self, tau):
        return self.t

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
        # caustic.q = proj.q[idx]
        # caustic.xi = proj.xi.iat[idx]
        S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)

    return S_F
