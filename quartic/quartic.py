# -*- coding: utf-8 -*-
"""
Configuration file for propagating a Gaussian in quartic potential using FINCO. It
contains the system's parameters and functions for the potential and initial
state, as well as a class for mapping between initial conditions and trajectories
in time and functions for dealing with nonphysical contributions.

Usage of the function dealing with nonphysical contributions can be seen in the
caustic exploration scripts in this folder. Refer to `caustics_illustration.py`
for usage example.
"""

import numpy as np
import pandas as pd

from finco import TimeTrajectory
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2

###########################
#    System parameters    #
###########################
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

#######################
#    Initial state    #
#######################
def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2 -
                (chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))

def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))

def S0_2(q):
    return np.full_like(q, 2j*gamma0)

S0 = [S0_0, S0_1, S0_2]

#######################
#      Potential      #
#######################
def V_0(q, _):
    return a*q**2 + b*q**4

def V_1(q, _):
    return 2*a*q + 4*b*q**3

def V_2(q, _):
    return 2*a + 12*b*q**2

V = [V_0, V_1, V_2]

#########################
#    Time trajectory    #
#########################
class QuarticTimeTrajectory(TimeTrajectory):
    """
    Time trajectory for propagating in quartic potential. In practice a simple
    propagation in real time.

    Parameters
    ----------
    T : float, optional
        Final propagation time. The default is 0.72.

    """
    def __init__(self, T = 0.72):
        self.T = T

    def init(self, ics):
        self.t = np.full_like(ics.q, self.T)

    def t_0(self, tau):
        return self.t * tau

    def t_1(self, _):
        return self.t

def eliminate_stokes(result):
    """
    Locates caustics and calculates factors for each trajectory to eliminate
    nonphysical contributions.

    Parameters
    ----------
    result : FINCOResults
        Results ataset to calculate for.

    Returns
    -------
    S_F : pandas.Series of float
        Factors for each trajectory in `result`, to eliminate nonphysical
        contributions.
    """
    # Load projection map, map to a grid, and calculate F
    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)

    # Plot the caustics map (map of xi_1). Uncomment if needed
    # plt.figure()
    # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

    blobs = separate_to_blobs(deriv, quantile=1e-2)
    qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]

    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2],
                             time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3)

    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (_, caustic) in caustics.iterrows():
        S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)

    return S_F
