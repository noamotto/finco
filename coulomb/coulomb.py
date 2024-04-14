#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for propagating the 1D Coulomb ground state using FINCO with
external field. It contains the system's parameters and functions for the potential
and initial state, as well as a class for mapping between initial conditions and
trajectories in time and functions for dealing with nonphysical contributions.

Remarks
-------

1. The coordinate q here is radial, hence the lack of absolute values.
2. This is still a prototype file, to be expanded as stable ways to propagate
trajectories and locate nonphysical trajectories are found for this system.
Currently only contains some basic time trajectory classes.
"""

import os
import logging

import numpy as np

from joblib import cpu_count

from finco.time_traj import SequentialTraj,LineTraj,CircleTraj
from finco.coord2time import Space2TimeTraj

m =1
keldysh = 1e4
omega = 7.35e-2
A0 = -omega / keldysh
q_e = 1
halfcycle = 2 * np.pi

# Logging
logging.basicConfig()

# Determine default number of jobs, as number of physical cores - 1 or
# using a designated environment variable
n_jobs = int(os.getenv('NCPUS', default = cpu_count(True) - 1))

def S0_0(q):
    return 1j * (q - np.log(q) - 0.5*np.log(2))

def S0_1(q):
    return np.array(1j * (1 - 1./q))

def S0_2(q):
    return 1j / q**2

def V_0(q, t):
    return -q_e / q + A0 * np.sin(t * omega) * np.exp(-(omega*t-np.pi/2)**2) * q

def V_1(q, t):
    return q_e / q**2 + A0 * np.sin(t * omega) * np.exp(-(omega*t-np.pi/2)**2)

def V_2(q, _):
    return -2 * q_e / q**3

S0 = [S0_0, S0_1, S0_2]
V = [V_0, V_1, V_2]

class PolesLookupTraj(SequentialTraj):
    """
    Trajectory in position space to lookup for poles.

    Based on the fact that in the problem we have 3 possible branches of position
    and momentum, and that choosing the radius of circling in position correcly
    results in a correct choice of circling poles in time.

    Follows a pattern of alternating between one big loop (which moves to the
    next pole) and two tight loops (stays on the same pole) to allow an estimation
    of the poles' locations.

    Parameters
    ----------
    n : int
        Number of poles to lookup.
    """
    def __init__(self, n):
        super().__init__(t0 = 0, t1 = 1)

        self.n = n

    def init(self, ics):
        def add_block(t0, t1):
            dt = t1 - t0

            self.path += ([CircleTraj(t0 = t0, t1 = t0 + 0.4*dt,
                                      a = self.r_1, r = self.r_1, turns = self.turns_1, phi0 = 0),
                           LineTraj(t0=t0 + 0.4*dt, t1=t0 + 0.5*dt, a=self.r_1, b = self.r_2),
                           CircleTraj(t0 = t0 + 0.5*dt, t1 = t0 + 0.9*dt,
                                      a = self.r_2, r = self.r_2, turns = self.turns_2, phi0 = 0),
                           LineTraj(t0=t0 + 0.9*dt, t1=t1, a=self.r_2, b = self.r_1)])
            self.discont_times += [t0, t0 + 0.4*dt, t0 + 0.5*dt, t0 + 0.9*dt]

        q0, p0 = ics.q.to_numpy(), ics.p.to_numpy()
        E0 = p0**2/2-1/q0
        self.r_n = (1.7 / np.abs(E0)) * q0 / np.abs(q0)
        self.r_k = (0.4 / np.abs(E0)) * q0 / np.abs(q0)

        # Determine whether first pole should be skipped. Done by doing the big loop first
        self.skip = (-V[1](q0,0) * m / p0 * -q0).real > 0
        self.r_1 = np.choose(self.skip, np.array([self.r_k, self.r_n]))
        self.turns_1 = np.choose(self.skip, np.array([2, 1]))
        self.r_2 = np.choose(self.skip, np.array([self.r_n, self.r_k]))
        self.turns_2 = 3 - self.turns_1

        ts = np.linspace(0.3/self.n, 1 - 0.3/self.n, self.n+1)
        self.path = [LineTraj(t0=0, t1=0.3/self.n, a=q0, b = self.r_1)]
        self.discont_times = []

        for (t0, t1) in zip(ts[:-1], ts[1:]):
            add_block(t0, t1)

        self.path.append(LineTraj(t0=1 - 0.3/self.n, t1=1, a = self.r_1, b = q0))
        self.discont_times.append(1-0.3/self.n)

        return self

    def get_beginning_times(self):
        """
        Returns the trajectory parameter markers for the beginning of each pole
        estimation.

        Returns
        -------
        Arraylike of shape (ntrajs, poles)
            The trajectory parameter markers for the beginning of each pole
        """
        return np.take([self.discont_times[0:-1:4], self.discont_times[2:-1:4]],
                       self.skip, axis=0)

    def get_ending_times(self):
        """
        Returns the trajectory parameter markers for the ending of each pole
        estimation.

        Returns
        -------
        Arraylike of shape (ntrajs, poles)
            The trajectory parameter markers for the beginning of each pole
        """
        return np.take([self.discont_times[1:-1:4], self.discont_times[3:-1:4]],
                       self.skip, axis=0)

class CirclingPolesTraj(SequentialTraj):
    """
    A general class for circling poles in time for the Coulomb problem

    The class works in 2 steps:

    1. The poles are found by building a lookup trajectory (using PolesLookupTraj)
    and estimating the poles locations
    2. The trajectory is built by moving from one pole to the other and circling them.

    Parameters
    ----------
    n : int
        Number of poles to estimate and circle.
    T : float
        Final time to propagate to.
    """
    def __init__(self, n, T):
        super().__init__(t0 = 0, t1 = 1)
        self.n = n
        self.T = T
        self.logger = logging.getLogger('coulomb.time_traj')

    def init(self, ics):
        def get_poles(ics):
            npoints = 200*self.n
            tt = Space2TimeTraj(t0=0, t1=1, q_traj=PolesLookupTraj(n=self.n),
                                V=V, m=m, max_step=1e-5).init(ics)
            taus = np.linspace(0, 1, npoints)
            ts = np.stack([tt.t_0(t) for t in taus]).T
            tbs = tt.q_traj.get_beginning_times()
            tes = tt.q_traj.get_ending_times()
            return np.array([[np.mean(ts_[int(tb_*npoints):int(te_*npoints)], axis=0)
                             for tb_, te_ in zip(tb,te)]
                             for ts_, tb,te in zip(ts, tbs,tes)])

        def add_block(t0, t1, a, pole):
            dt = t1 - t0
            self.path += ([LineTraj(t0=t0, t1=t0 + 0.5*dt, a=a, b = pole - self.r*self.sgn),
                           CircleTraj(t0 = t0 + 0.5*dt, t1 = t1,
                                      a = pole - self.r*self.sgn, r = -self.r*self.sgn, turns = 1,
                                      phi0 = 0)])
            self.discont_times += [t0 + 0.5*dt, t1]

        self.logger.debug('Looking for %d poles for %d initial conditions', self.n, len(ics))
        self.poles = get_poles(ics)
        # Estimate one radius based on the difference between poles
        self.r = np.mean(np.abs(np.diff(self.poles)), axis=1) / 2 * 1j

        # build trajectories
        self.logger.debug('Building trajectories for circling %d poles for %d initial conditions',
                          self.n, len(ics))
        self.sgn = np.sign(self.poles[:,0].imag)
        self.a = self.poles - (self.r*self.sgn)[:,np.newaxis]
        self.path = []
        self.discont_times = []

        ts = np.linspace(0, 1-0.5/self.n, self.n+1)
        cur = np.zeros(self.poles.shape[0])
        for (t0, t1, pole) in zip(ts[:-1], ts[1:], self.poles.T):
            add_block(t0, t1, cur, pole)
            cur = pole - self.r*self.sgn

        self.path.append(LineTraj(t0=1-0.5/self.n, t1=1, a=self.poles[:,-1] - self.r*self.sgn,
                                  b = self.T))
