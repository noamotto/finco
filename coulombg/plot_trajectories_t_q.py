# -*- coding: utf-8 -*-
"""
Propagates and plots the trajectory in time and position spaces for one initial
condition and three different "orders" of circling poles in time, ending after
one cycle. It is recommended to enlarge and use tight layout after the figure
is produced.
"""

#%% Setup
import os

from coulombg import V, S0, m, CoulombGTimeTrajectory, coulombg_pole, halfcycle

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics

plt.rc('font', size=14)

def plot_markers(x, ax, diff, **kwargs):
    for point in np.arange(diff, len(x), diff):
        direction = np.angle(x[point+1] - x[point]) / np.pi * 180
        ax.plot(np.real(x[point]), np.imag(x[point]), marker=(3,1,direction - 90), ms=7, **kwargs)

T = 1*2*halfcycle
qs = np.array([-1+1j])
ics = create_ics(qs, S0 = S0, gamma_f=1)

try:
    os.mkdir('system_exploration')
except FileExistsError:
    pass

fig, ((t_o0, t_o1, t_o2), (q_o0, q_o1, q_o2)) = plt.subplots(2, 3,
                                                             figsize=(14.4, 9.6))
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
    n = np.arange(-5, 6, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o0.set_title('n=0')
    t_o0.scatter(tstar.real, tstar.imag, c=n)
    t_o0.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=2)
    plot_markers(t[:600].to_numpy(), t_o0, 150, color='r')
    t_o0.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'b', lw=2)
    plot_markers(t[600:1200].to_numpy(), t_o0, 60, color='b')
    t_o0.set_xlabel(r'$\Re t$')
    t_o0.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_o0.scatter([0], [0])
    q_o0.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=2)
    plot_markers(q[:600].to_numpy(), q_o0, 150, color='r', lw=2)
    q_o0.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'b', lw=2)
    plot_markers(q[600:1200].to_numpy(), q_o0, 60, color='b')
    q_o0.set_xlabel(r'$\Re q$')
    q_o0.set_ylabel(r'$\Im q$')


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
    n = np.arange(-5, 6, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o1.set_title('n=1')
    t_o1.scatter(tstar.real, tstar.imag, c=n)
    t_o1.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=2)
    plot_markers(t[:600].to_numpy(), t_o1, 150, color='r')
    t_o1.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'g', lw=2)
    plot_markers(t[600:1200].to_numpy(), t_o1, 60, color='g')
    t_o1.plot(np.real(t[1200:1800]), np.imag(t[1200:1800]), 'b', lw=2)
    plot_markers(t[1200:1800].to_numpy(), t_o1, 60, color='b')
    t_o1.set_xlabel(r'$\Re t$')
    t_o1.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_o1.scatter([0], [0])
    q_o1.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=2)
    plot_markers(q[:600].to_numpy(), q_o1, 150, color='r')
    q_o1.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'g', lw=2)
    plot_markers(q[600:1200].to_numpy(), q_o1, 60, color='g')
    q_o1.plot(np.real(q[1200:1800]), np.imag(q[1200:1800]), 'b', lw=2)
    plot_markers(q[1200:1800].to_numpy(), q_o1, 60, color='b')
    q_o1.set_xlabel(r'$\Re q$')
    q_o1.set_ylabel(r'$\Im q$')

#%% Order 2

order = 4
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
    n = np.arange(-5, 6, 1)
    tstar = coulombg_pole(q0, p0, n)

    t_o2.set_title('n=2')
    t_o2.scatter(tstar.real, tstar.imag, c=n)
    t_o2.plot(np.real(t[:600]),      np.imag(t[:600]), 'r', lw=2)
    plot_markers(t[:600].to_numpy(), t_o2, 150, color='r')
    t_o2.plot(np.real(t[600:1200]),  np.imag(t[600:1200]), 'g', lw=2)
    plot_markers(t[600:1200].to_numpy(), t_o2, 60, color='g')
    t_o2.plot(np.real(t[1200:1800]), np.imag(t[1200:1800]), 'c', lw=2)
    plot_markers(t[1200:1800].to_numpy(), t_o2, 150, color='c')
    t_o2.plot(np.real(t[1800:2400]), np.imag(t[1800:2400]), 'm', lw=2)
    plot_markers(t[1800:2400].to_numpy(), t_o2, 60, color='m')
    t_o2.plot(np.real(t[2400:3000]), np.imag(t[2400:3000]), 'b', lw=2)
    plot_markers(t[2400:3000].to_numpy(), t_o2, 100, color='b')
    t_o2.set_xlabel(r'$\Re t$')
    t_o2.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_o2.scatter([0], [0])
    q_o2.plot(np.real(q[:600]),      np.imag(q[:600]), 'r', lw=2)
    plot_markers(q[:600].to_numpy(), q_o2, 150, color='r')
    q_o2.plot(np.real(q[600:1200]),  np.imag(q[600:1200]), 'g', lw=2)
    plot_markers(q[600:1200].to_numpy(), q_o2, 60, color='g')
    q_o2.plot(np.real(q[1200:1800]), np.imag(q[1200:1800]), 'c', lw=2)
    plot_markers(q[1200:1800].to_numpy(), q_o2, 100, color='c')
    q_o2.plot(np.real(q[1800:2400]), np.imag(q[1800:2400]), 'm', lw=2)
    plot_markers(q[1800:2400].to_numpy(), q_o2, 60, color='m')
    q_o2.plot(np.real(q[2400:3000]), np.imag(q[2400:3000]), 'b', lw=2)
    plot_markers(q[2400:3000].to_numpy(), q_o2, 100, color='b')
    q_o2.set_xlabel(r'$\Re q$')
    q_o2.set_ylabel(r'$\Im q$')

fig.tight_layout()
fig.savefig('system_exploration/trajs_q_f.png')