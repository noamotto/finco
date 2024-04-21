#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts for testing how trajectories work in Coulomb without external field.
The tests mainly compare the method from coulombg of building trajectories and
a simple method of following poles based on their location.
This tests should be used to make sure things work after changing the trajectory
building code.

The first cell produces a scatter plot of the trajectory, with the analytical
poles in red and the estimated in blue.
The second cell propagates both a trajectory circling the anytical and estimated
poles, and compares the distance between final results. In the case of no external
field we expect the propagation around the analytical poles to provide the
reference value.

@author: Noam Ottolenghi
"""
#%% Setup

import logging

import numpy as np
import matplotlib.pyplot as plt

from finco import create_ics, propagate
from finco.coord2time import Space2TimeTraj
from coulomb import S0, m, halfcycle, PolesLookupTraj, CirclingPolesTraj
from coulombg import coulombg_pole, CoulombGTimeTrajectory, V as V_nofield

q0 = np.array([15-15j, 15+15j])
p0 = S0[1](q0)
ics = create_ics(q0, S0)

n = 10
T = halfcycle*2*3

logging.basicConfig()
logging.getLogger('coulomb').setLevel(logging.DEBUG)

#%% Comparison test between estimated pole locations and analytical locations

tt = Space2TimeTraj(t0=0, t1=1, q_traj=PolesLookupTraj(n=n), V=V_nofield,
                    m=m, max_step=1e-5).init(ics)

taus = np.linspace(0, 1, 1200)
ts = np.stack([tt.t_0(t) for t in taus])
plt.figure()
plt.scatter(np.real(ts[:,0]), np.imag(ts[:,0]), c=np.arange(1200))

n_star = np.arange(-n - 1, 1, 1)
tstar = coulombg_pole(q0[0], p0[0], n_star)
plt.scatter(tstar.real, tstar.imag, c='r')

tbs = tt.q_traj.get_beginning_times()[0]
tes = tt.q_traj.get_ending_times()[0]
tstar_e = np.array([np.mean(ts[int(tb*1200):int(te*1200),0]) for tb,te in zip(tbs,tes)])
plt.scatter(tstar_e.real, tstar_e.imag, c='b')

#%% Compare the resulting trajectories between coulombg and circumnavigating the found poles

tt_e = CirclingPolesTraj(n=n, T=T)
result_e = propagate(ics, V = V_nofield, m = m, gamma_f = 1,
                     time_traj = tt_e,
                     dt = 1e-5, drecord=1/1200,
                     n_jobs = 1, blocksize=2**10,
                     trajs_path=None)
res_e = result_e.get_results()

result_s = propagate(ics, V = V_nofield, m = m, gamma_f = 1,
                     time_traj = CoulombGTimeTrajectory(n=n, k=0),
                     dt = 1e-5, drecord=1/1200,
                     n_jobs = 1, blocksize=2**10,
                     trajs_path=None)
res_s = result_s.get_results()

diff = np.max(np.abs(res_e.iloc[-1].to_numpy() - res_s.iloc[-1].to_numpy()))
print(f'Maximal parameter difference between estimated and analytical: {diff}')
