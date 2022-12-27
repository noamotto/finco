# -*- coding: utf-8 -*-
"""
Propagates and plots the trajectory in time and position spaces for one initial
condition and three different "orders" of circling poles in time, ending after
one cycle. It is recommended to enlarge and use tight layout after the figure
is produced.
"""

#%% Setup

from coulombg import V, S0, m, CoulombGTimeTrajectory, coulombg_pole, halfcycle

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics

T = 1*2*halfcycle
qs = np.array([3-3j])
ics = create_ics(qs, S0 = S0, gamma_f=1)

fig, ((t_o0, t_o1, t_o2), (q_o0, q_o1, q_o2)) = plt.subplots(2, 3)
fig.suptitle(f'q={qs[0]}, T={T:.3f}')

#%% Order 0

order=0
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=T),
                   dt = 1e-4, drecord=1/1200, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    n = np.arange(-10, 11, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o0.set_title('n=0')
    t_o0.scatter(tstar.real, tstar.imag, c=n)
    t_o0.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=3)
    t_o0.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'b', lw=3)

    # Plot space trajectory
    q_o0.scatter([0], [0])
    q_o0.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=3)
    q_o0.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'b', lw=3)


#%% Order 1

order = 1
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=T),
                   dt = 1e-4, drecord=1/1800, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    n = np.arange(-10, 11, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o1.set_title('n=1')
    t_o1.scatter(tstar.real, tstar.imag, c=n)
    t_o1.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=3)
    t_o1.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'g', lw=3)
    t_o1.plot(np.real(t[1200:1800]), np.imag(t[1200:1800]), 'b', lw=3)

    # Plot space trajectory
    q_o1.scatter([0], [0])
    q_o1.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=3)
    q_o1.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'g', lw=3)
    q_o1.plot(np.real(q[1200:1800]), np.imag(q[1200:1800]), 'b', lw=3)

#%% Order 2

order = 2
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=T),
                   dt = 1e-4, drecord=1/3000, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    n = np.arange(-10, 11, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o2.set_title('n=2')
    t_o2.scatter(tstar.real, tstar.imag, c=n)
    t_o2.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=3)
    t_o2.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'g', lw=3)
    t_o2.plot(np.real(t[1200:1800]), np.imag(t[1200:1800]), 'c', lw=3)
    t_o2.plot(np.real(t[1800:2400]), np.imag(t[1800:2400]), 'm', lw=3)
    t_o2.plot(np.real(t[2400:3000]), np.imag(t[2400:3000]), 'b', lw=3)

    # Plot space trajectory
    q_o2.scatter([0], [0])
    q_o2.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=3)
    q_o2.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'g', lw=3)
    q_o2.plot(np.real(q[1200:1800]), np.imag(q[1200:1800]), 'c', lw=3)
    q_o2.plot(np.real(q[1800:2400]), np.imag(q[1800:2400]), 'm', lw=3)
    q_o2.plot(np.real(q[2400:3000]), np.imag(q[2400:3000]), 'b', lw=3)
