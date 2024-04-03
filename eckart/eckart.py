# -*- coding: utf-8 -*-
"""
Configuration file for propagating a Gaussian in Eckart using FINCO. It
contains the system's parameters and functions for the potential and initial
state, as well as a class for mapping between initial conditions and trajectories
in time.

TODO: Due to the branch-cuts of the hyperbolic functions in the potential the
calculations of poles in time can have errors that influence the resulting
reconstruction. Those were not checked and solved here as this configuration
file was not used for reconstruction. If you consider using this configuration
file for reconstructions you should check and fix those errors.
"""

#%% Setup

import os
import logging

import numpy as np
from joblib import cpu_count

from finco.time_traj import SequentialTraj, LineTraj

# Logging
logging.basicConfig()

# Determine default number of jobs, as number of physical cores - 1 or
# using a designated environment variable
n_jobs = int(os.getenv('NCPUS', default = cpu_count(True) - 1))

###########################
#    System parameters    #
###########################
m = 1060
V0 = 0.01562
a = 0.734

#######################
#    Initial state    #
#######################
q_c = -8 + 0j
gamma_c = 0.5 + 0j
p_c = 4 + 0j

def S0_0(q):
    return 1j*gamma_c*(q - q_c)**2 + p_c * (q - q_c) - 1j/4 * np.log(2 * gamma_c / np.pi)

def S0_1(q):
    return 2j*gamma_c*(q - q_c) + p_c

def S0_2(q):
    return np.full_like(q, 2j*gamma_c)

S0 = [S0_0, S0_1, S0_2]

#######################
#      Potential      #
#######################
def V_0(q, _):
    return V0 / np.cosh(q / a) ** 2

def V_1(q, _):
    return -2 * V0 / a * np.sinh(q / a) / np.cosh(q / a) ** 3

def V_2(q, _):
    return -2 * V0 / a ** 2 / np.cosh(q / a) ** 4 * (1 - 2 * np.sinh(q / a) ** 2)

V = [V_0, V_1, V_2]

###########################
#    Pole calcualation    #
###########################
def eckart_pole(q0, p0, n=0, sign=0):
    """
    Calculates the location in time for a pole in Eckart potential given initial
    conditions.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions
    p0 : ArrayLike of complex
        Initial momenta
    n : integer, optional
        Pole number on the ladder, where 0 matches roughly the pole closest to
        the real axis. The default is 0.
    sign : either 0 or 1, optional
        Which pole ladder to calculate from, where 0 corresponds to the ladder
        with the bigger real part (the "right" ladder). The default is 0.

    Returns
    -------
    tstars : ArrayLike of complex
        The calculated poles in temporal space.
    """
    E0 = p0**2/2/m + V_0(q0, 0)
    v = (2 * E0/m) ** 0.5
    s = (-1)**sign
    return a / v * (-np.arctanh(p0/m/v / np.tanh(q0 / a)) +
                    s * np.arctanh((E0 / V0)**0.5) +
                    1j * np.pi * (n + 0.5))

def eckart_diff(q0, p0):
    """
    Calculates the difference in time between adjecant poles in Eckart potential
    given initial conditions.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions
    p0 : ArrayLike of complex
        Initial momenta

    Returns
    -------
    diffs : list of 2 complex
        The differences in time. diffs[0] corresponds to the difference between
        adjecant poles on the same ladder, and diffs[1] corresponds to the
        difference between matching poles on the two ladders.
    """
    return (eckart_pole(q0, p0, n=1) - eckart_pole(q0, p0, n=0),
            eckart_pole(q0, p0, sign=0) - eckart_pole(q0, p0, sign=1))

#########################
#    Time trajectory    #
#########################
class EckartTimeTrajectory(SequentialTraj):
    """
    Time trajectory for propagating in Eckart barrier, assuming we start on the
    left of the barrier (negative real part).

    The built trajectory consists of 7 parts:
    1. approaching the first ("left") poles ladder
    2. Moving between the two poles marking our "entry point", before entering
    the pole ladders
    3. Moving into the middle between the pole ladders
    4. Moving between the pole ladders up to the two poles marking our "exit point"
    5. Moving out of the second ("right") pole ladder
    6. Moving back to the real axis
    7. Propagating to final time

    The code handles edge cases in which one or more of the paths above is merged
    with another.

    Parameters
    ----------
    n0 : integer
        Pole number on the "left" ladder above which is the "entry point"
    n1 : integer
        Pole number on the "right" ladder above which is the "exit point"
    t : Callable or float
        Final propagation time. If float, then one time is taken for all initial
        conditions. If not, should be a callable that accepts an ArrayLike of
        initial positions and an ArrayLike of initial positions and returns an
        ArrayLike of the same size with the final propagation times.
    """

    def __init__(self, n0, n1, t):
        super().__init__(t0 = 0, t1 = 1)

        self.n0 = n0
        self.n1 = n1

        if isinstance(t, float):
            self.t = lambda q,p: np.full_like(q, t)
        else:
            self.t = t

    def init(self, ics):
        # Calc Radii of circles
        q0, p0, t0 = ics.q0.to_numpy(), ics.p0.to_numpy(), ics.t.to_numpy()
        diff_y, diff_x = eckart_diff(q0, p0)
        # Calc points of interest on the trajectory

        # t0: Initial point
        t0 = t0 if t0 is not None else np.zeros_like(q0)

        # t1: Final point
        t1 = self.t(q0, p0)

        # a: Point approaching the poles line
        self.a = np.array((eckart_pole(q0, p0, n=-1, sign=1) +
                           eckart_pole(q0, p0, n=0, sign=1) -
                           diff_x) / 2)

        # b: Point of entrance to the poles ladder
        self.b = self.a + self.n0*diff_y

        # c: Point in the middle of the ladder
        self.c = self.b + diff_x

        # d: Point between the poles in the exit point
        self.d = self.c + (self.n1 - self.n0)*diff_y

        # e: Point approaching out of the ladder
        self.e = self.d + diff_x

        # f: Point back on real axis
        self.f = self.e - self.n1*diff_y

        # Build path
        if self.n0 != 0:
            if self.n0 != self.n1:
                if self.n1 != 0:
                    self.path = [LineTraj(t0=0,   t1=1/7, a=t0,     b=self.a),
                                 LineTraj(t0=1/7, t1=2/7, a=self.a, b=self.b),
                                 LineTraj(t0=2/7, t1=3/7, a=self.b, b=self.c),
                                 LineTraj(t0=3/7, t1=4/7, a=self.c, b=self.d),
                                 LineTraj(t0=4/7, t1=5/7, a=self.d, b=self.e),
                                 LineTraj(t0=5/7, t1=6/7, a=self.e, b=self.f),
                                 LineTraj(t0=6/7, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 7) / 7
                else: # e = f
                    self.path = [LineTraj(t0=0,   t1=1/6, a=t0,     b=self.a),
                                 LineTraj(t0=1/6, t1=2/6, a=self.a, b=self.b),
                                 LineTraj(t0=2/6, t1=3/6, a=self.b, b=self.c),
                                 LineTraj(t0=3/6, t1=4/6, a=self.c, b=self.d),
                                 LineTraj(t0=4/6, t1=5/6, a=self.d, b=self.e),
                                 LineTraj(t0=5/6, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 6) / 6
            else: # c = d
                if self.n1 != 0:
                    self.path = [LineTraj(t0=0,   t1=1/6, a=t0,     b=self.a),
                                 LineTraj(t0=1/6, t1=2/6, a=self.a, b=self.b),
                                 LineTraj(t0=2/6, t1=3/6, a=self.b, b=self.c),
                                 LineTraj(t0=3/6, t1=4/6, a=self.d, b=self.e),
                                 LineTraj(t0=4/6, t1=5/6, a=self.e, b=self.f),
                                 LineTraj(t0=5/6, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 6) / 6
                else: # e = f
                    self.path = [LineTraj(t0=0,   t1=1/5, a=t0,     b=self.a),
                                 LineTraj(t0=1/5, t1=2/5, a=self.a, b=self.b),
                                 LineTraj(t0=2/5, t1=3/5, a=self.b, b=self.c),
                                 LineTraj(t0=3/5, t1=4/5, a=self.d, b=self.e),
                                 LineTraj(t0=4/5, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 5) / 5
        else: # a = b
            if self.n0 != self.n1:
                if self.n1 != 0:
                    self.path = [LineTraj(t0=0,   t1=1/6, a=t0,     b=self.a),
                                 LineTraj(t0=1/6, t1=2/6, a=self.b, b=self.c),
                                 LineTraj(t0=2/6, t1=3/6, a=self.c, b=self.d),
                                 LineTraj(t0=3/6, t1=4/6, a=self.d, b=self.e),
                                 LineTraj(t0=4/6, t1=5/6, a=self.e, b=self.f),
                                 LineTraj(t0=5/6, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 6) / 6
                else: # e = f
                    self.path = [LineTraj(t0=0,   t1=1/5, a=t0,     b=self.a),
                                 LineTraj(t0=1/5, t1=2/5, a=self.b, b=self.c),
                                 LineTraj(t0=2/5, t1=3/5, a=self.c, b=self.d),
                                 LineTraj(t0=3/5, t1=4/5, a=self.d, b=self.e),
                                 LineTraj(t0=4/5, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 5) / 5
            else: # c = dsinh
                if self.n1 != 0:
                    self.path = [LineTraj(t0=0,   t1=1/5, a=t0,     b=self.a),
                                 LineTraj(t0=1/5, t1=2/5, a=self.b, b=self.c),
                                 LineTraj(t0=2/5, t1=3/5, a=self.d, b=self.e),
                                 LineTraj(t0=3/5, t1=4/5, a=self.e, b=self.f),
                                 LineTraj(t0=4/5, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 5) / 5
                else: # e = f
                    self.path = [LineTraj(t0=0,   t1=1/4, a=t0,     b=self.a),
                                 LineTraj(t0=1/4, t1=2/4, a=self.b, b=self.c),
                                 LineTraj(t0=2/4, t1=3/4, a=self.d, b=self.e),
                                 LineTraj(t0=3/4, t1=1,   a=self.f, b=t1),]
                    self.discont_times = np.arange(1, 4) / 4

        self.discont_times = list(self.discont_times)
        return self
