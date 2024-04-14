# -*- coding: utf-8 -*-
"""
File with miscellaneous sketches. Will probably be removed in future commits.
"""

#%% Setup
from coulombg import V, S0, m, CoulombGTimeTrajectory, n_jobs, coulombg_pole, coulombg_diff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from finco import propagate, create_ics
from utils import tripcolor_complex

#%%
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

#%% Trajectory crash test
    
# X, Y = np.meshgrid(np.linspace(1e-10, 5, 302), np.linspace(-4, 4, 302))
# X, Y = np.meshgrid(np.linspace(0, 10, 41), np.linspace(-6, 6, 49))
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
order=2
# qs = qs[(qs != 1j) & (qs != -1j)]
# qs = np.array([(1.6692307692973847-0.8544698544698544j),
#                 (1.6769230769895385-0.8544698544698544j)])
# qs = np.array(-3+3j)

def crash_t(q0, p0):
    return coulombg_pole(q0, p0, np.sign(coulombg_diff(q0, p0).real) * order)

ics = create_ics(qs, S0 = S0, gamma_f=1)
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=crash_t),
                   dt = 1e-4, drecord=1, n_jobs=3)

trajs = result.get_trajectories(1).sort_index()
poles = coulombg_pole(trajs.q0, trajs.p0, 0)
diffs = coulombg_diff(trajs.q0, trajs.p0)
probs = pd.Series(np.imag(poles) - np.imag(diffs) / np.real(diffs)*np.real(poles) < 0)

plt.figure(), plt.scatter(np.real(trajs.q0), np.imag(trajs.q0), c=trajs.q.apply(lambda x: 'b' if np.abs(x) < 1e-1 else 'r'))
plt.figure(), plt.scatter(np.real(trajs.q0), np.imag(trajs.q0), c=probs.apply(lambda x: 'r' if x else 'b'))

#%%
def crash(q,p):
    t = CoulombGTimeTrajectory(n=0).init(create_ics(q,S0,gamma_f=1))
    return coulombg_pole(q,p,t.nfirst + k)
q = np.array([-1 - 4j])
ics = create_ics(q, S0 = S0, gamma_f=1)
plt.figure(fr'$q={q}$')

for k in np.arange(-3,4):
    result = propagate(ics, V = V, m = m, gamma_f = 1,
                            time_traj = CoulombGTimeTrajectory(n=0, t=crash),
                            dt = 1e-3, drecord=1/100,
                            n_jobs = n_jobs, blocksize=2**10,
                            trajs_path=None)
    plt.scatter(np.real(result.t.loc[:,50:]), np.imag(result.t.loc[:,50:]),
                c=np.real(result.p.loc[:, 50:]))
    
#%%
def crash(q,p):
    t = CoulombGTimeTrajectory(n=n).init(create_ics(q,S0,gamma_f=1))
    return t.b + t.u

n=6
X, Y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-6, 6, 300))
qs = (X+1j*Y)[(Y != 1) & (Y != -1)]
ics = create_ics(qs, S0 = S0, gamma_f=1)
result = propagate(ics, V = V, m = m, gamma_f = 1,
                        time_traj = CoulombGTimeTrajectory(n=n, t=crash),
                        dt = 1e-4, drecord=1/100,
                        n_jobs = n_jobs, blocksize=2**10,
                        trajs_path=None)