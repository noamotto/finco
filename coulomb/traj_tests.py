#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:24:28 2024

@author: nomyle
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

from coulomb import S0, V, m, halfcycle, PolesLookupTraj, CirclingPolesTraj
from coulombg import coulombg_pole, CoulombGTimeTrajectory

from finco import create_ics, propagate
from finco.coord2time import Space2TimeTraj

#%%

q0 = np.array([15-15j, 15+15j])
p0 = S0[1](q0)
ics = create_ics(q0, S0)

n = 6
T = halfcycle*2*3

logging.getLogger('coulomb').setLevel(logging.DEBUG)
tt = Space2TimeTraj(t0=0, t1=1, q_traj=PolesLookupTraj(n=n), V=V, m=m, max_step=1e-5).init(ics)

taus = np.linspace(0, 1, 1200)
ts = np.stack([tt.t_0(t) for t in taus])
plt.figure(), plt.scatter(np.real(ts[:,0]), np.imag(ts[:,0]), c=np.arange(1200))

n_star = np.arange(-n - 1, 1, 1)
tstar = coulombg_pole(q0[0], p0[0], n_star)
plt.scatter(tstar.real, tstar.imag, c='r')

tbs = tt.q_traj.get_beginning_times()[0]
tes = tt.q_traj.get_ending_times()[0]
tstar_e = np.array([np.mean(ts[int(tb*1200):int(te*1200),0]) for tb,te in zip(tbs,tes)])
plt.scatter(tstar_e.real, tstar_e.imag, c='b')

#%%
tt_e = CirclingPolesTraj(n=n, T=T)
result_e = propagate(ics, V = V, m = m, gamma_f = 1,
                     time_traj = tt_e,
                     dt = 1e-4, drecord=1/1200,
                     n_jobs = 1, blocksize=2**10,
                     trajs_path=None)
res_e = result_e.get_results()

result_s = propagate(ics, V = V, m = m, gamma_f = 1,
                     time_traj = CoulombGTimeTrajectory(n=n, k=0),
                     dt = 1e-4, drecord=1/1200,
                     n_jobs = 1, blocksize=2**10,
                     trajs_path=None)
res_s = result_s.get_results()

#%%
from finco.time_traj import SequentialTraj, LineTraj, CircleTraj
from finco.coord2time import Space2TimeTraj

q_e = 1
A0 = 1e-6
def V_0(q, t):
    return -q_e / q + A0 * q

def V_1(q, t):
    return q_e / q**2 + A0

def V_2(q, t):
    return -2 * q_e / q**3

V = [V_0, V_1, V_2]

q0 = np.array([1-1j])
p0 = np.array([1+1j])
ics = create_ics(q0, S0)
ics.p0 = p0[0]
ics.p = p0[0]
E0 = p0**2 / 2 / m + V[0](q0, 0)
rs = np.concatenate([(-E0+(E0**2-4*A0)**0.5)/2/A0, (-E0-(E0**2-4*A0)**0.5)/2/A0])
rs = rs[np.argsort(np.abs(rs))]
rs = np.abs(rs) * q0 / np.abs(q0)

class TestTraj(SequentialTraj):
    def __init__(self, r):
        super().__init__(t0=0, t1=1)
        self.r = r
        
    def init(self, ics):
        q0 = ics.q.to_numpy()
        r = np.abs(self.r / q0) * q0
        self.path = [LineTraj(t0=0, t1=0.3, a=q0, b=r),
                     CircleTraj(t0=0.3, t1=1, a=r, r=r, turns=1, phi0=0)]
        self.discont_times = [0.3]
        
        return self
        
tt1 = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(rs[0]*1.1), V=V, m=m, max_step=1e-5).init(ics)
# r2 = np.mean(np.abs(rs)) * rs[0] / np.abs(rs[0])
# tt2 = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r2), V=V, m=m, max_step=1e-5).init(ics)
# tt3 = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(rs[1]/0.9), V=V, m=m, max_step=1e-5).init(ics)

