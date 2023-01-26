# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

import os

import numpy as np
import pandas as pd
import logging
from joblib import cpu_count

from finco.time_traj import TimeTrajectory, CircleTraj, LineTraj
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, approximate_F

# Logging
logging.basicConfig()

# Determine default number of jobs, as number of physical cores - 1 or
# using a designated environment variable
n_jobs = int(os.getenv('NCPUS', default = cpu_count(True) - 1))

# System params
m = 1060
V0 = 0.01562
a = 0.734

q_c = -8 + 0j
gamma_c = 0.5 + 0j
p_c = 4 + 0j

def S0_0(q):
    return 1j*gamma_c*(q - q_c)**2 + p_c * (q - q_c) - 1j/4 * np.log(2 * gamma_c / np.pi)

def S0_1(q):
    return 2j*gamma_c*(q - q_c) + p_c

def S0_2(q):
    return np.full_like(q, 2j*gamma_c)

def V_0(q):
    return V0 / np.cosh(q / a) ** 2

def V_1(q):
    return -2 * V0 / a * np.sinh(q / a) / np.cosh(q / a) ** 3

def V_2(q):
    return -2 * V0 / a ** 2 / np.cosh(q / a) ** 4 * (1 - 2 * np.sinh(q / a) ** 2)

V = [V_0, V_1, V_2]
S0 = [S0_0, S0_1, S0_2]

def eckart_pole(q0, p0, n=0, sign=0):
    E0 = p0**2/2/m + V_0(q0)
    v = (2 * E0/m) ** 0.5
    s = (-1)**sign
    return a / v * (-np.arctanh(p0/m/v / np.tanh(q0 / a)) + 
                    s * np.arctanh((E0 / V0)**0.5) + 
                    1j * np.pi * (n + 0.5))

def eckart_diff(q0, p0):
    return (eckart_pole(q0, p0, n=1) - eckart_pole(q0, p0, n=0), 
            eckart_pole(q0, p0, sign=0) - eckart_pole(q0, p0, sign=1))


class EckartTimeTrajectory(TimeTrajectory):
    def __init__(self, n0, n1, t):
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
            else: # c = d
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

    def t_0(self, tau):
        ts = np.array(self.discont_times + [1])
        path = np.flatnonzero(ts >= tau)[0]
        return self.path[path].t_0(tau)

    def t_1(self, tau):
        ts = np.array(self.discont_times + [1])
        path = np.flatnonzero(ts >= tau)[0]
        return self.path[path].t_1(tau)

    def get_discontinuity_times(self):
        return self.discont_times

# def locate_caustics(result, n, t, n_jobs=1):
#     logger = logging.getLogger('analysis.stokes')

#     deriv = result.get_caustics_map(1)

#     # plt.figure()
#     # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

#     blobs = separate_to_blobs(deriv, quantile=1e-2, connectivity=3)
#     qs = [deriv.q0[deriv.xi_1.abs()[list(blob)].idxmin()] for blob in blobs]

#     logger.info('Found {} caustic candidates'.format(len(qs)))
#     logger.debug('located caustic candidates at\n {}'.format(qs))
#     caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2],
#                              time_traj=CoulombGTimeTrajectory(n), dt=1e-4, gamma_f=1,
#                              n_jobs=n_jobs)
    
#     caustics = caustics[np.real(caustics.q) > 0]

#     logger.info('Caustic root search yielded {} caustics'.format(len(caustics)))
#     logger.debug('located caustics at\n {}'.format(caustics.q.to_numpy()))

#     return caustics

# def eliminate_stokes(result, caustics, sigma=None):
#     logger = logging.getLogger('analysis.stokes')
    
#     # Filter the small caustics. 50 Works well here as a magic number...
#     # caustics = caustics[np.abs(caustics.xi_2) < 50]
    
#     deriv = result.get_caustics_map(1)
#     proj = result.get_projection_map(1)
    
#     if sigma is None:
#         sigma = proj.sigma

#     S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
#     for (i, caustic) in caustics.iterrows():
#         logger.debug('handling caustic at {}'.format(caustic.q))
#         s_f = calc_factor2(caustic, proj.q0, proj.xi, sigma)
#         F, _ = approximate_F(proj.q0, proj.xi, caustic)
#         r = np.abs(-caustic.xi_2*2/caustic.xi_3)
#         s_f[np.abs(F.v_t) > r] = 1
#         S_F *= s_f
#         # plt.figure(), plt.tripcolor(np.real(proj.q0), np.imag(proj.q0), S_F), plt.colorbar()
#         # plt.scatter(np.real(caustic.q), np.imag(caustic.q))
#     S_F *= (np.real(proj.sigma) <= 0)
#     S_F *= (np.abs(deriv.xi_1) <= 100)

#     return S_F

