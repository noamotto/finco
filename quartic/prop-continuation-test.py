# -*- coding: utf-8 -*-
"""
Test of validity of propagation contuation procedure.

The main goal of the test, aside of showcasing how to use continue_propagation(),
is to assure the the propagation is continued correctly. 

Two cases are compared here, both for a gaussian in quartic potential. In the
first one full propagation takes place with two snapshots, and in the second it
is broken into two parts, each with one snapshot taken. The snapshots are then
compared to make sure the values are close.

@author: Noam Ottolenghi
"""

import numpy as np

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import propagate, continue_propagation, TimeTrajectory, create_ics

class PartialTimeTrajectory(TimeTrajectory):
    def __init__(self, traj, t0, t1):
        self.traj = traj
        self.t0 = t0
        self.t1 = t1
        
    def init(self, ics):
        self.traj.init(ics)
        
        return self

    def t_0(self, tau):
        t = tau * (self.t1 - self.t0) + self.t0
        return self.traj.t_0(t)

    def t_1(self, tau):
        t = tau * (self.t1 - self.t0) + self.t0
        dt = self.t1 - self.t0
        return self.traj.t_1(t) * dt
    
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 121), np.linspace(-2.5, 2.5, 121))
qs = (X+1j*Y).flatten()
jac = (X[0,1] - X[0,0]) *  (Y[1,0] - Y[0,0])

ics = create_ics(qs, S0 = S0)
traj = QuarticTimeTrajectory()
results1 = propagate(ics, V = V, m = m, gamma_f=1,
                     time_traj = traj, dt = 1e-3, drecord=1/2,
                     blocksize=1024, n_jobs=3, trajs_path=None)

results2 = propagate(ics, V = V, m = m, gamma_f=1,
                     time_traj = PartialTimeTrajectory(traj=traj, t0=0, t1=0.5),
                     dt = 1e-3, drecord=1, blocksize=1024, n_jobs=3, trajs_path=None)

results3 = continue_propagation(results2, V = V, m = m, gamma_f=1,
                                time_traj = PartialTimeTrajectory(traj=traj, t0=0.5, t1=1),
                                dt = 1e-3, drecord = 1, blocksize=1024, n_jobs=3, trajs_path=None)

res1_1 = results1.get_results(1,2)
res1_2 = results1.get_results(2)
res2_1 = results2.get_results(1)
res2_2 = results3.get_results(1)

print(f'First half fits: {np.all(np.isclose(np.abs(res1_1.to_numpy() - res2_1.to_numpy()), 0))}')
print(f'Second half fits: {np.all(np.isclose(np.abs(res1_2.to_numpy() - res2_2.to_numpy()), 0))}')
