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

halfcycle = 2 * np.pi

def S0_0(q):
    return 1j * (q - np.log(q) - 0.5*np.log(2))

def S0_1(q):
    return np.array(1j * (1 - 1./q))

def S0_2(q):
    return 1j / q**2

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


class EckartTrajectory(TimeTrajectory):
    def __init__(self, n, t = 3*2*halfcycle):
        self.n = n
        
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

        # a: Point of entrance to the poles line.
        self.a = np.array(np.mean(eckart_pole(q0, p0, np.array([-1,0,-1,0]), np.array([1,1,0,0]))) - diff_x)
        self.a[(np.imag(q0) <= 0) & (np.real(q0) >= 0)] += self.r[(np.imag(q0) <= 0) & (np.real(q0) >= 0)] *2
        # self.a -= self.r *2
        # self.first[q0.real < 0] -= 1

        # b: Point of exit from the poles line
        self.b = self.a + 2*self.n*self.r
        
        # c: Point for enough from ploes line, towards the ending position
        self.c = np.array([self.b - self.r * 1j, self.b + self.r * 1j])
        close = np.argmin(np.abs(self.c - t1)[:,np.newaxis], axis=0)
        self.c = np.take_along_axis(self.c, close, axis=0).squeeze()

        # Build path
        self.path = []
        self.discont_times = []
        if self.n==0:
            self.discont_times = [1/2, 3/4]

            self.path.append(LineTraj(t0=0, t1=1/2, a=t0, b=self.a))
            self.path.append(LineTraj(t0=1/2, t1=3/4, a=self.b, b=self.c))
            self.path.append(LineTraj(t0=3/4, t1=1, a=self.c, b=t1))

        elif self.n==1:
            self.discont_times = [1/3, 2/3, 5/6]

            self.path.append(LineTraj(t0=0, t1=1/3, a=t0, b=self.a))
            self.path.append(CircleTraj(t0=1/3, t1=2/3, a=self.a, r=self.r,
                                        turns=-1.5 + self.first, phi0=np.pi))
            self.path.append(LineTraj(t0=2/3, t1=5/6, a = self.b, b=self.c))
            self.path.append(LineTraj(t0=5/6, t1=1, a = self.c, b=t1))

        else:
            Ts = list(np.linspace(1/(2*self.n+1), 1-1/(2*self.n+1), 2*self.n)) + [1-1/(2*self.n+1)/2]
            self.discont_times = Ts

            self.path.append(LineTraj(t0=0, t1=Ts[0], a=t0, b=self.a))
            self.path.append(CircleTraj(t0=Ts[0], t1=Ts[1], a=self.a, r=self.r,
                                        turns=-1.25 + self.first, phi0=np.pi))
            self.path.append(LineTraj(t0=Ts[1], t1=Ts[2],
                                      a=self.a + (1+1j)*self.r, b=self.a + (3+1j)*self.r))

            for i in range(self.n-2):
                a = self.a + (2*i+3+1j)*self.r
                self.path.append(CircleTraj(t0=Ts[2*i+2], t1=Ts[2*i+3],
                                            a=a, r=self.r, turns=-1, phi0=np.pi/2))
                self.path.append(LineTraj(t0=Ts[2*i+3], t1=Ts[2*i+4], a=a, b=a+2*self.r))

            self.path.append(CircleTraj(t0=Ts[-3], t1=Ts[-2],
                                        a=self.a + (2*self.n-1+1j)*self.r, r=self.r,
                                        turns=-1.25, phi0=np.pi/2))
            self.path.append(LineTraj(t0=Ts[-2], t1=Ts[-1], a=self.b, b=self.c))
            self.path.append(LineTraj(t0=Ts[-1], t1=1, a=self.c, b=t1))

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
