# -*- coding: utf-8 -*-
"""
Configuration file for propagating the 1D Coulomb ground state using FINCO. It
contains the system's parameters and functions for the potential and initial
state, as well as a class for mapping between initial conditions and trajectories
in time and functions for dealing with nonphysical contributions.

Remarks
-------

1. The coordinate q here is radial, hence the lacking of absolute values.
2. The class for mapping between initial conditions and trajectories in time
requires a parameter at construction. This is the number of poles to circumnavigate
in time, or "order" of the system. It also allows to enter the final propagation
time for the trajectories.
3. The usage of the functions dealing with nonphysical contributions can be seen
in various exploration and analysis scripts in this folder. Refer to `caustic_times.py`
for usage of finding caustic times, and to `analyze_results.py` for an example of
using locate_caustics() and eliminate_stokes().
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
m = 1
keldysh = 1
omega = 7.35e-2
A0 = -omega / keldysh
q_e = 1

halfcycle = 2 * np.pi

def S0_0(q):
    return 1j * (q - np.log(q) - 0.5*np.log(2))

def S0_1(q):
    return np.array(1j * (1 - 1./q))

def S0_2(q):
    return 1j / q**2

def V_0(q):
    return -q_e / q

def V_1(q):
    return q_e / q**2

def V_2(q):
    return -2 * q_e / q**3

V = [V_0, V_1, V_2]
S0 = [S0_0, S0_1, S0_2]

def coulombg_pole(q0, p0, n=0):
    E0 = p0**2/2/m - 1/q0
    sign = np.sign(q0.real)*np.sign(q0.imag)
    return (-q0*p0/2/E0 + (m/2)**0.5 * (np.log((2*E0/m)**0.5*sign*p0*q0 + 2*E0*q0 + 1)
                                        + n*1j*np.pi*2)/2/E0**1.5/sign)

def coulombg_diff(q0, p0):
    return coulombg_pole(q0, p0, n=1) - coulombg_pole(q0, p0, n=0)


class CoulombGTimeTrajectory(TimeTrajectory):
    def __init__(self, n, t = 3*2*halfcycle):
        self.n = n
        
        if isinstance(t, float):
            self.t = lambda q,p: np.full_like(q, t)
        else:
            self.t = t

    def init(self, ics):
        # Calc trajectory parameters
        q0, p0, t0 = ics.q0.to_numpy(), ics.p0.to_numpy(), ics.t.to_numpy()
        
        # Calc the 2 directions for the trajectory
        # r: in the direction of the poles line, towards positive time
        #   with length of the circle's radius
        # u: perpendicular to r, with the same length, in the direction
        #   pointing to the poles line from the origin
        self.r = np.array(-coulombg_diff(q0, p0)) / 2
        self.r *= np.sign(self.r.real)
        self.u = (coulombg_pole(q0, p0, n=0) - 
                  (np.real(coulombg_pole(q0, p0, n=0)*self.r.conj()) * self.r /
                   np.abs(self.r)**2))
        self.u *= np.abs(self.r) / np.abs(self.u)
        
        # Calc whether u points "above" the poles line or "below". Needed to
        # calculate the circle's entry point and to calculate which trajectories
        # need additional rotation.
        self.dir = np.sign((self.u*self.r.conj()).imag)
        self.first = np.zeros(q0.shape)
        # self.first[(q0.real < 0) & (self.dir < 0)] -= 1
        

        # Calc points of reference on the trajectory

        # t0: Initial point
        t0 = t0 if t0 is not None else np.zeros_like(q0)
        
        # t1: Final point
        t1 = self.t(q0, p0)

        # a: Point starting to circle poles
        self.nfirst = np.zeros(q0.shape)
        self.nfirst[(q0.imag < 0) & (self.dir > 0)] -= 1
        self.a = coulombg_pole(q0, p0, n=self.nfirst) - self.u
        self.turns = np.ones(q0.shape) 
        self.turns[(q0.real < 0) & (self.dir < 0)] += 1
        # self.a[q0.imag < 0] -= self.r *2

        # b: Point of exit from the poles line
        self.b = self.a + ((self.n - 1) * 2)*self.r

        # Build path
        self.path = []
        self.discont_times = []
        if self.n==0:
            self.discont_times = [1/2]

            self.path.append(LineTraj(t0=0, t1=1/2, a=t0, b=self.a))
            self.path.append(LineTraj(t0=1/2, t1=1, a=self.a, b=t1))

        elif self.n==1:
            self.discont_times = [1/3, 2/3]

            self.path.append(LineTraj(t0=0, t1=1/3, a=t0, b=self.a))
            self.path.append(CircleTraj(t0=1/3, t1=2/3, a=self.a, r=self.r,
                                        turns=self.turns, phi0=-np.pi/2*self.dir))
            self.path.append(LineTraj(t0=2/3, t1=1, a = self.a, b=t1))

        else:
            Ts = list(np.linspace(1/(2*self.n+1), 1-1/(2*self.n+1), 2*self.n))
            self.discont_times = Ts

            self.path.append(LineTraj(t0=0, t1=Ts[0], a=t0, b=self.a))
            self.path.append(CircleTraj(t0=Ts[0], t1=Ts[1], a=self.a, r=self.r,
                                        turns=1, phi0=-np.pi/2*self.dir))
            self.path.append(LineTraj(t0=Ts[1], t1=Ts[2],
                                      a=self.a, b=self.a + 2*self.r))

            for i in range(self.n-2):
                a = self.a + 2*(i+1)*self.r
                self.path.append(CircleTraj(t0=Ts[2*i+2], t1=Ts[2*i+3],
                                            a=a, r=self.r, turns=self.turns, phi0=-np.pi/2*self.dir))
                self.path.append(LineTraj(t0=Ts[2*i+3], t1=Ts[2*i+4], a=a, b=a + 2*self.r))

            self.path.append(CircleTraj(t0=Ts[-2], t1=Ts[-1],
                                        a=self.b, r=self.r,
                                        turns=3, phi0=-np.pi/2*self.dir))
            self.path.append(LineTraj(t0=Ts[-1], t1=1, a=self.b, b=t1))

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

def locate_caustics(result, n, t, n_jobs=1):
    logger = logging.getLogger('analysis.stokes')

    deriv = result.get_caustics_map(1)

    # plt.figure()
    # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

    blobs = separate_to_blobs(deriv, quantile=1e-2, connectivity=3)
    qs = [deriv.q0[deriv.xi_1.abs()[list(blob)].idxmin()] for blob in blobs]

    logger.info('Found {} caustic candidates'.format(len(qs)))
    logger.debug('located caustic candidates at\n {}'.format(qs))
    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2],
                             time_traj=CoulombGTimeTrajectory(n), dt=1e-4, gamma_f=1,
                             n_jobs=n_jobs)
    
    caustics = caustics[np.real(caustics.q) > 0]

    logger.info('Caustic root search yielded {} caustics'.format(len(caustics)))
    logger.debug('located caustics at\n {}'.format(caustics.q.to_numpy()))

    return caustics

def eliminate_stokes(result, caustics, sigma=None):
    logger = logging.getLogger('analysis.stokes')
    
    # Filter the small caustics. 50 Works well here as a magic number...
    # caustics = caustics[np.abs(caustics.xi_2) < 50]
    
    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)
    
    if sigma is None:
        sigma = proj.sigma

    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (i, caustic) in caustics.iterrows():
        logger.debug('handling caustic at {}'.format(caustic.q))
        s_f = calc_factor2(caustic, proj.q0, proj.xi, sigma)
        F, _ = approximate_F(proj.q0, proj.xi, caustic)
        r = np.abs(-caustic.xi_2*2/caustic.xi_3)
        s_f[np.abs(F.v_t) > r] = 1
        S_F *= s_f
        # plt.figure(), plt.tripcolor(np.real(proj.q0), np.imag(proj.q0), S_F), plt.colorbar()
        # plt.scatter(np.real(caustic.q), np.imag(caustic.q))
    S_F *= (np.real(proj.sigma) <= 0)
    S_F *= (np.abs(deriv.xi_1) <= 100)

    return S_F

def coulombg_caustic_times_dist(q0, p0, t0, est):    
    if est:
        dt = 1e-2
    else:
        dt = 1e-2
            
    r = np.array(-coulombg_diff(np.array(q0), np.array(p0)) / 2)
    dts = np.abs(r) * dt
    dts[np.abs(r) > 1] = dt
    return dts

def coulombg_caustic_times_dir(q0 ,p0 ,t0, est):
    r = np.array(-coulombg_diff(np.array(q0), np.array(p0)) / 2)
    direction = np.array([t0 - r * 1j, t0 + r * 1j])
    close = np.argmin(np.abs(direction - t0)[:,np.newaxis], axis=0)
    direction = np.take_along_axis(direction, close, axis=0).squeeze()
    return direction / np.abs(direction)
