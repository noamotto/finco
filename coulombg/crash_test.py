# -*- coding: utf-8 -*-
"""
Crash test for time trajectories in Coulomb ground state with no external field.
This test is useful to determine which trajectories did manage to circle poles
correctly.

The test propagates the trajectories to the last circled pole in time, then plots
which trajectories did crash (blue) and which didn't (red).

@author: Noam Ottolenghi
"""

#%% Setup
from coulombg import V, S0, m, CoulombGTimeTrajectory, n_jobs

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics

#%% Run
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
order = 2

def crash_t(q0, _):
    """
    Determines crash times for trajectories. Used as function to determine final
    times for each trajectory. Determined using CoulombGTimeTrajectory's intermediate
    parameters created when initializing.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Trajectory initial position.
    p0 : ArrayLike of complex
        Trajectory initial momentum.

    Returns
    -------
    t : ArrayLike of complex
        Trajectory crash times.
    """
    t = CoulombGTimeTrajectory(n=order).init(create_ics(q0,S0))
    return t.b + t.u

ics = create_ics(qs, S0 = S0)
result = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=crash_t),
                   dt = 1e-4, drecord=1, n_jobs=n_jobs, trajs_path=None)

trajs = result.get_trajectories(1)

plt.figure()
plt.scatter(np.real(trajs.q0), np.imag(trajs.q0),
            c=trajs.q.apply(lambda x: 'b' if np.abs(x) < 1e-1 else 'r'))
