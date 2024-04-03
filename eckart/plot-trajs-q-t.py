# -*- coding: utf-8 -*-
"""
Propagates and plots the trajectory in time and position spaces for one initial
condition and four different configurations of entering and exiting the poles
ladder for propagation in Eckart barrier potential. It is recommended to enlarge
and use tight layout after the figure is produced.
"""

#%% Setup

import numpy as np

from eckart import q_c, p_c, S0, V, m, a, EckartTimeTrajectory, eckart_pole
from finco import propagate, create_ics
import matplotlib.pyplot as plt

plt.rc('font', size=14)

def plot_markers(x, ax, diff, **kwargs):
    """
    Adds arrow markers on plot with calculated direction

    Parameters
    ----------
    x : ArrayLike of complex
        Points on plot to put markers for. Used for calculation of position and
        direction.
    ax : matplotlib Axes
        Axes to plot the markers into
    diff : integer
        Space between two markers. Marker is plotted once every `diff` points

    All other parameters are passed to Axes.plot()
    """
    for point in np.arange(diff, len(x), diff):
        direction = np.angle(x[point+1] - x[point]) / np.pi * 180
        ax.plot(np.real(x[point]), np.imag(x[point]), marker=(3,1,direction - 90), ms=10, **kwargs)

T = 5500.
qs = np.array([q_c])
ics = create_ics(qs, S0 = S0)

fig, ((t_o0, t_o1, t_o2, t_o3),
      (q_o0, q_o1, q_o2, q_o3)) = plt.subplots(2, 4, num='trajs-q-f', figsize=(14.4, 9.6))

#%%
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = EckartTimeTrajectory(n0=0, n1=0, t=T),
                   dt = 1e-3, drecord=1/1200, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    M, S = np.meshgrid(np.arange(-2,3), [1,0])
    t_poles = eckart_pole(q_c, p_c, M, S)

    t_o0.scatter(t_poles.real, t_poles.imag)
    t_o0.plot(np.real(t[:300]),      np.imag(t[:300]), 'r', lw=2)
    plot_markers(t[:300].to_numpy(), t_o0, 75, color='r')
    t_o0.plot(np.real(t[300:600]),   np.imag(t[300:600]), 'g', lw=2)
    plot_markers(t[300:600].to_numpy(), t_o0, 150, color='g')
    t_o0.plot(np.real(t[600:900]),   np.imag(t[600:900]), 'c', lw=2)
    plot_markers(t[600:900].to_numpy(), t_o0, 150, color='c')
    t_o0.plot(np.real(t[900:1200]), np.imag(t[900:1200]), 'b', lw=2)
    plot_markers(t[900:1200].to_numpy(), t_o0, 60, color='b')
    t_o0.set_xlabel(r'$\Re t$')
    t_o0.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_poles = -1j * a * (2*np.arange(-2,3)+1)/2*np.pi
    q_o0.scatter(q_poles.real, q_poles.imag)
    q_o0.plot(np.real(q[:300]),      np.imag(q[:300]), 'r', lw=2)
    plot_markers(q[:300].to_numpy(), q_o0, 75, color='r')
    q_o0.plot(np.real(q[300:600]),   np.imag(q[300:600]), 'g', lw=2)
    plot_markers(q[300:600].to_numpy(), q_o0, 150, color='g')
    q_o0.plot(np.real(q[600:900]),   np.imag(q[600:900]), 'c', lw=2)
    plot_markers(q[600:900].to_numpy(), q_o0, 200, color='c')
    q_o0.plot(np.real(q[900:1200]), np.imag(q[900:1200]), 'b', lw=2)
    plot_markers(q[900:1200].to_numpy(), q_o0, 60, color='b')
    q_o0.set_xlabel(r'$\Re q$')
    q_o0.set_ylabel(r'$\Im q$')

#%%
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = EckartTimeTrajectory(n0=1, n1=0, t=T),
                   dt = 1e-3, drecord=1/1200, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    M, S = np.meshgrid(np.arange(-2,3), [1,0])
    t_poles = eckart_pole(q_c, p_c, M, S)

    t_o1.scatter(t_poles.real, t_poles.imag)
    t_o1.plot(np.real(t[:200]),      np.imag(t[:200]), 'r', lw=2)
    plot_markers(t[:200].to_numpy(), t_o1, 50, color='r')
    t_o1.plot(np.real(t[200:400]),   np.imag(t[200:400]), 'g', lw=2)
    plot_markers(t[200:400].to_numpy(), t_o1, 50, color='g')
    t_o1.plot(np.real(t[400:600]),   np.imag(t[400:600]), 'c', lw=2)
    plot_markers(t[400:600].to_numpy(), t_o1, 100, color='c')
    t_o1.plot(np.real(t[600:800]),   np.imag(t[600:800]), 'm', lw=2)
    plot_markers(t[600:800].to_numpy(), t_o1, 50, color='m')
    t_o1.plot(np.real(t[800:1000]),  np.imag(t[800:1000]), 'k', lw=2)
    plot_markers(t[800:1000].to_numpy(), t_o1, 100, color='k')
    t_o1.plot(np.real(t[1000:1200]),  np.imag(t[1000:1200]), 'b', lw=2)
    plot_markers(t[1000:1200].to_numpy(), t_o1, 50, color='b')
    t_o1.set_xlabel(r'$\Re t$')
    t_o1.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_poles = -1j * a * (2*np.arange(-2,3)+1)/2*np.pi
    q_o1.scatter(q_poles.real, q_poles.imag)
    q_o1.plot(np.real(q[:200]),      np.imag(q[:200]), 'r', lw=2)
    plot_markers(q[:200].to_numpy(), q_o1, 50, color='r')
    q_o1.plot(np.real(q[200:400]),   np.imag(q[200:400]), 'g', lw=2)
    plot_markers(q[200:400].to_numpy(), q_o1, 50, color='g')
    q_o1.plot(np.real(q[400:600]),   np.imag(q[400:600]), 'c', lw=2)
    plot_markers(q[400:600].to_numpy(), q_o1, 100, color='c')
    q_o1.plot(np.real(q[600:800]),   np.imag(q[600:800]), 'm', lw=2)
    plot_markers(q[600:800].to_numpy(), q_o1, 100, color='m')
    q_o1.plot(np.real(q[800:1000]),  np.imag(q[800:1000]), 'k', lw=2)
    plot_markers(q[800:1000].to_numpy(), q_o1, 100, color='k')
    q_o1.plot(np.real(q[1000:1200]),  np.imag(q[1000:1200]), 'b', lw=2)
    plot_markers(q[1000:1200].to_numpy(), q_o1, 50, color='b')
    q_o1.set_xlabel(r'$\Re q$')
    q_o1.set_ylabel(r'$\Im q$')

#%%
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = EckartTimeTrajectory(n0=0, n1=1, t=T),
                   dt = 1e-3, drecord=1/1200, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    M, S = np.meshgrid(np.arange(-2,3), [1,0])
    t_poles = eckart_pole(q_c, p_c, M, S)

    t_o2.scatter(t_poles.real, t_poles.imag)
    t_o2.plot(np.real(t[:200]),      np.imag(t[:200]), 'r', lw=2)
    plot_markers(t[:200].to_numpy(), t_o2, 50, color='r')
    t_o2.plot(np.real(t[200:400]),   np.imag(t[200:400]), 'g', lw=2)
    plot_markers(t[200:400].to_numpy(), t_o2, 100, color='g')
    t_o2.plot(np.real(t[400:600]),   np.imag(t[400:600]), 'c', lw=2)
    plot_markers(t[400:600].to_numpy(), t_o2, 50, color='c')
    t_o2.plot(np.real(t[600:800]),   np.imag(t[600:800]), 'm', lw=2)
    plot_markers(t[600:800].to_numpy(), t_o2, 100, color='m')
    t_o2.plot(np.real(t[800:1000]),  np.imag(t[800:1000]), 'k', lw=2)
    plot_markers(t[800:1000].to_numpy(), t_o2, 50, color='k')
    t_o2.plot(np.real(t[1000:1200]),  np.imag(t[1000:1200]), 'b', lw=2)
    plot_markers(t[1000:1200].to_numpy(), t_o2, 50, color='b')
    t_o2.set_xlabel(r'$\Re t$')
    t_o2.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_poles = -1j * a * (2*np.arange(-2,3)+1)/2*np.pi
    q_o2.scatter(q_poles.real, q_poles.imag)
    q_o2.plot(np.real(q[:200]),      np.imag(q[:200]), 'r', lw=2)
    plot_markers(q[:200].to_numpy(), q_o2, 50, color='r')
    q_o2.plot(np.real(q[200:400]),   np.imag(q[200:400]), 'g', lw=2)
    plot_markers(q[200:400].to_numpy(), q_o2, 100, color='g')
    q_o2.plot(np.real(q[400:600]),   np.imag(q[400:600]), 'c', lw=2)
    plot_markers(q[400:600].to_numpy(), q_o2, 100, color='c')
    q_o2.plot(np.real(q[600:800]),   np.imag(q[600:800]), 'm', lw=2)
    plot_markers(q[600:800].to_numpy(), q_o2, 100, color='m')
    q_o2.plot(np.real(q[800:1000]),  np.imag(q[800:1000]), 'k', lw=2)
    plot_markers(q[800:1000].to_numpy(), q_o2, 50, color='k')
    q_o2.plot(np.real(q[1000:1200]),  np.imag(q[1000:1200]), 'b', lw=2)
    plot_markers(q[1000:1200].to_numpy(), q_o2, 50, color='b')
    q_o2.set_xlabel(r'$\Re q$')
    q_o2.set_ylabel(r'$\Im q$')

#%%
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = EckartTimeTrajectory(n0=1, n1=-1, t=T),
                   dt = 1e-3, drecord=1/1400, n_jobs=1)

trajs = result.get_trajectories(1)
for q0, traj in trajs.groupby('q0'):
    p0 = S0[1](q0)
    t = traj.t
    q = traj.q

    # plot time trajectory
    E0 = p0**2/2/m - 1/q0
    M, S = np.meshgrid(np.arange(-2,3), [1,0])
    t_poles = eckart_pole(q_c, p_c, M, S)

    t_o3.scatter(t_poles.real, t_poles.imag)
    t_o3.plot(np.real(t[:200]),      np.imag(t[:200]), 'r', lw=2)
    plot_markers(t[:200].to_numpy(), t_o3, 50, color='r')
    t_o3.plot(np.real(t[200:400]),   np.imag(t[200:400]), 'g', lw=2)
    plot_markers(t[200:400].to_numpy(), t_o3, 50, color='g')
    t_o3.plot(np.real(t[400:600]),   np.imag(t[400:600]), 'c', lw=2)
    plot_markers(t[400:600].to_numpy(), t_o3, 100, color='c')
    t_o3.plot(np.real(t[600:800]),   np.imag(t[600:800]), 'm', lw=2)
    plot_markers(t[600:800].to_numpy(), t_o3, 40, color='m')
    t_o3.plot(np.real(t[800:1000]),  np.imag(t[800:1000]), 'k', lw=2)
    plot_markers(t[800:1000].to_numpy(), t_o3, 100, color='k')
    t_o3.plot(np.real(t[1000:1200]),  np.imag(t[1000:1200]), 'C1', lw=2)
    plot_markers(t[1000:1200].to_numpy(), t_o3, 50, color='C1')
    t_o3.plot(np.real(t[1200:1400]),  np.imag(t[1200:1400]), 'b', lw=2)
    plot_markers(t[1200:1400].to_numpy(), t_o3, 50, color='b')
    t_o3.set_xlabel(r'$\Re t$')
    t_o3.set_ylabel(r'$\Im t$')

    # Plot space trajectory
    q_poles = -1j * a * (2*np.arange(-2,3)+1)/2*np.pi
    q_o3.scatter(q_poles.real, q_poles.imag)
    q_o3.plot(np.real(q[:200]),      np.imag(q[:200]), 'r', lw=2)
    plot_markers(q[:200].to_numpy(), q_o3, 50, color='r')
    q_o3.plot(np.real(q[200:400]),   np.imag(q[200:400]), 'g', lw=2)
    plot_markers(q[200:400].to_numpy(), q_o3, 80, color='g')
    q_o3.plot(np.real(q[400:600]),   np.imag(q[400:600]), 'c', lw=2)
    plot_markers(q[400:600].to_numpy(), q_o3, 120, color='c')
    q_o3.plot(np.real(q[600:800]),   np.imag(q[600:800]), 'm', lw=2)
    plot_markers(q[600:800].to_numpy(), q_o3, 40, color='m')
    q_o3.plot(np.real(q[800:1000]),  np.imag(q[800:1000]), 'k', lw=2)
    plot_markers(q[800:1000].to_numpy(), q_o3, 120, color='k')
    q_o3.plot(np.real(q[1000:1200]),  np.imag(q[1000:1200]), 'C1', lw=2)
    plot_markers(q[1000:1200].to_numpy(), q_o3, 70, color='C1')
    q_o3.plot(np.real(q[1200:1400]),  np.imag(q[1200:1400]), 'b', lw=2)
    plot_markers(q[1200:1400].to_numpy(), q_o3, 50, color='b')
    q_o3.set_xlabel(r'$\Re q$')
    q_o3.set_ylabel(r'$\Im q$')

#%%
plt.tight_layout()
plt.savefig('complex-time-example.png')
