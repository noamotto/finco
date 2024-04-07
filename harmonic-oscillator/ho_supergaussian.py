# -*- coding: utf-8 -*-
"""
Configuration file for propagating a supergaussian of the shape :math:`~e^{-x^4}`
in harmonic potential using FINCO. It contains the system's parameters and functions
for the potential and initial state, as well as a class for mapping between initial
conditions and trajectories in time.
"""

import numpy as np

from finco import TimeTrajectory

###########################
#    System parameters    #
###########################
m = 1
omega = 1

#######################
#    Initial state    #
#######################
def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**4)

def S0_1(q):
    return 4j *(q - 1)**3

def S0_2(q):
    return 12j *(q - 1)**2

S0 = [S0_0, S0_1, S0_2]

###################
#    Potential    #
###################
def V_0(q, _):
    return 0.5 * m * omega**2 * q ** 2

def V_1(q, _):
    return m * omega**2 * q

def V_2(q, _):
    return np.full_like(q,m * omega**2)

V = [V_0, V_1, V_2]

#########################
#    Time Trajectory    #
#########################
class HOTimeTrajectory(TimeTrajectory):
    """
    Time trajectory for propagating in harmonic potential. In practice a simple
    propagation in real time.

    Parameters
    ----------
    T : float, optional
        Final propagation time. The default is pi*2.

    """
    def __init__(self, T = np.pi*2):
        self.T = T

    def init(self, ics):
        self.q0 = ics.q

    def t_0(self, tau):
        return np.full_like(self.q0, self.T*tau)

    def t_1(self, _):
        return np.full_like(self.q0, self.T)

    def get_discontinuity_times(self):
        return []
