# -*- coding: utf-8 -*-
"""
Produces a figure showing the Riemann sheets for the real part of momentum as a
function of time.

The script runs a group of trajectories with the same initial conditions and slightly
different trajectory in time to produce a rectangular cover in temporal space around
the two first poles, circling them.
Then a 3D graph is produced using mayavi2, where the x,y axes are real and imaginary
time respectively, and the z axis is the real part of momentum (those axes are
not written as it messes up the figure). The plot contains a depiction of the
first two Riemann sheets of the system. The first Riemann sheet which contains
circling the first pole is in blue, and the second sheet which contains circling
the second pole is in orange.
A red line is drawn on the two sheets depicting one trajectory in time and how it
propagates on the sheets, along with its projection on the temporal plane (xy plane).

A much simpler, flat version of the plot is also available, using matplotlib. The
plot contains the two sheets in two separate figures. The plot is on the temporal
plane, and the axes are the real and imaginary parts of time. The real part of
momentum is depicted using colormap. Because of that the "flat version" is quite
difficult to read, and it is advised not to use it. Its main advatage is that it
does not use mayavi which might be hard to install. The use the flat version,
uncomment the lines under the comment "Flat version", comment those under
"3D version", and comment the last cell.
"""

#%% Setup
import os

from coulombg import V, S0, m, coulombg_pole, coulombg_diff

import numpy as np
import matplotlib.pyplot as plt

# 3D Version
from mayavi import mlab
from matplotlib.tri import Triangulation

from finco import propagate, create_ics
from finco.time_traj import SequentialTraj, LineTraj

class InitialPlaneTimeTrajectory(SequentialTraj):
    """
    Time trajectory for drawing the first Riemann sheet for Coluomb potential
    without external field. Takes the initial conditions and draws different
    trajectories, given a parameter alpha deciding where to pass between the
    poles in the ladder. Follows a simple, rectangular shape.

    Parameters
    ----------
    alphas : ArrayLike of float in [0,1]
        Where to pass between the poles in the ladder. Should have one value for
        each trajectory that would be initialized, where alphas[i]=0 means the
        ith trajectory passes through pole number 2, and alphas[i]=1 means the
        ith trajectory passes through pole number -1. Should have the same length
        as the number of trajectories passed to init().
    """
    def __init__(self, alphas):
        super().__init__(t0=0, t1=1)
        self.alphas = alphas

    def init(self, ics):
        q0, p0 = ics.q0.to_numpy(), ics.p0.to_numpy()
        diff = coulombg_diff(q0, p0)
        poles_dir = -diff/np.abs(diff)
        diff_factor = (1/3-np.abs(self.alphas))*3

        t_f = (coulombg_pole(q0, p0, n=-1)*(1-self.alphas)/2 +
               coulombg_pole(q0, p0, n=2)*(self.alphas+1)/2)
        a = np.real(poles_dir * np.conj(t_f)) * poles_dir
        b = t_f + (t_f - a) * diff_factor
        c = b + np.sign(self.alphas) * diff * 1.5 * diff_factor
        d = t_f + + np.sign(self.alphas) * diff * 1.5 * diff_factor

        mask = np.abs(self.alphas) > 1/3
        b[mask] = 2/3*a[mask] + 1/3*t_f[mask]
        c[mask] = 1/3*a[mask] + 2/3*t_f[mask]
        d[mask] = t_f[mask]

        self.path = [LineTraj(t0=0,     t1=0.25,    a=0, b=a),
                     LineTraj(t0=0.25,  t1=0.5,     a=a, b=b),
                     LineTraj(t0=0.5,   t1=0.75,    a=b, b=c),
                     LineTraj(t0=0.75,  t1=1,       a=c, b=d)]
        self.discont_times = [0.25, 0.5, 0.75]
        return self

class SecondPlaneTimeTrajectory(SequentialTraj):
    """
    Time trajectory for drawing the second Riemann sheet for Coluomb potential
    without external field. Takes the initial conditions and draws different
    trajectories, first passing them between the two first poles to advance them
    to the next sheet, given a parameter alpha deciding where to pass between the
    poles in the ladder. Follows a rectangular shape.

    Parameters
    ----------
    alphas : ArrayLike of float in [0,1]
        Where to pass between the poles in the ladder. Should have one value for
        each trajectory that would be initialized, where alphas[i]=0 means the
        ith trajectory passes through pole number 1 to advance to the next sheet,
        and alphas[i]=1 means the ith trajectory passes through pole number 0 to
        advance to the next sheet. Alpha then depicts where to pass in thesecond
        sheet to form it. Should have the same length as the number of trajectories
        passed to init().
    """
    def __init__(self, alphas):
        super().__init__(t0=0, t1=1)
        self.alphas = alphas

    def init(self, ics):
        q0, p0 = ics.q0.to_numpy(), ics.p0.to_numpy()
        diff = coulombg_diff(q0, p0)
        poles_dir = -diff/np.abs(diff)

        # First sheet
        t_inner = (coulombg_pole(q0, p0, n=0)*(1-self.alphas)/2 +
                   coulombg_pole(q0, p0, n=1)*(self.alphas+1)/2)
        a = np.real(poles_dir * np.conj(t_inner)) * poles_dir
        b = t_inner + (t_inner - a) * (self.alphas+1)/2
        c = b - diff * 2 * (self.alphas+1) / 2
        d = t_inner - diff * 2 * (self.alphas+1)/2

        # Second sheet
        e = d - (t_inner - a) * (self.alphas+1)/2
        f = e + 2.2 * diff * (self.alphas+1)/2
        g = f + (t_inner - a) * (self.alphas+1)
        h = g - diff - 2 * diff * (self.alphas+1)/2
        i = t_inner - diff - 2 * diff * (self.alphas+1)/2

        self.path = [LineTraj(t0=0.0,  t1=0.05, a=0, b=a),
                     LineTraj(t0=0.05, t1=0.1,  a=a, b=b),
                     LineTraj(t0=0.1,  t1=0.15, a=b, b=c),
                     LineTraj(t0=0.15, t1=0.2,  a=c, b=d),
                     LineTraj(t0=0.2,  t1=0.36, a=d, b=e),
                     LineTraj(t0=0.36, t1=0.52, a=e, b=f),
                     LineTraj(t0=0.52, t1=0.68, a=f, b=g),
                     LineTraj(t0=0.68, t1=0.84, a=g, b=h),
                     LineTraj(t0=0.84, t1=1.0,  a=h, b=i)]
        self.discont_times = [0.05, 0.1, 0.15, 0.2, 0.36, 0.52, 0.68, 0.84]

        return self

try:
    os.mkdir('system_exploration')
except FileExistsError:
    pass

#%% Initial Plane

q = 2+0.5j
alphas = np.concatenate([np.linspace(-1, -1/3-1e-2, 20),
                        np.linspace(-1/3+1e-2, 1/3-1e-2, 40),
                        np.linspace(1/3-1e-2, 1, 20)])
qs = np.full(len(alphas), q)
ics = create_ics(qs, S0 = S0)

result1 = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj=InitialPlaneTimeTrajectory(alphas=alphas),
                   dt=1e-3, drecord=1/400, trajs_path=None)

# Flat version

# trajs1 = result1.get_trajectories(start=0, end=400)angle
# plt.figure()
# plt.title(f'q={q}, first sheet'.format(q))
# plt.scatter(np.real(trajs1.t), np.imag(trajs1.t), c=np.real(trajs1.p))
# plt.colorbar()

# 3D version

mlab.figure('riemann_sheets', size=(900, 500))

parts = [0,20,60,80]
trajs1 = result1.get_trajectories(start=100, end=400)
for start, end in zip(parts[:-1], parts[1:]):
    trajs = trajs1.loc[start:end]
    triangles = Triangulation(np.real(trajs.t), np.imag(trajs.t)).triangles
    mlab.triangular_mesh(np.real(trajs.t), np.imag(trajs.t),
                          np.real(trajs.p), triangles, color=plt.cm.tab10(0)[:3])

#%% Second plane
alphas = np.linspace(-0.97, 0.97, 80)
qs = np.full(len(alphas), q)
ics = create_ics(qs, S0 = S0)

result2 = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj=SecondPlaneTimeTrajectory(alphas=alphas),
                   dt=1e-3, drecord=1/500, trajs_path=None)

# Flat version

# trajs2 = result2.get_trajectories(start=100, end=500)
# plt.figure()
# plt.title(f'q={q}, second sheet')
# plt.scatter(np.real(trajs2.t), np.imag(trajs2.t), c=np.real(trajs2.p))
# plt.colorbar()

# 3D version

times = SecondPlaneTimeTrajectory(alphas=alphas).init(ics).get_discontinuity_times()[4:] + [1]
for start, end in zip(times[:-1], times[1:]):
    trajs2 = result2.get_trajectories(start=start*500, end=end*500)
    triangles = Triangulation(np.real(trajs2.t), np.imag(trajs2.t)).triangles
    mlab.triangular_mesh(np.real(trajs2.t), np.imag(trajs2.t),
                          np.real(trajs2.p), triangles, color=plt.cm.tab10(1)[:3])

traj = result2.get_trajectories(start=0, end=500).loc[50]
mlab.plot3d(np.real(traj.t), np.imag(traj.t), np.real(traj.p)+0.1, color=(1,0,0))
mlab.plot3d(np.real(traj.t), np.imag(traj.t), np.linspace(-4, -3.7, len(traj.t)), color=(1,0,0))

#%% Scene stuff
view = (-75, 63, 31,
        np.array([ 1.8,  1.55, -2.0]))

mlab.axes(color=(1,1,1), extent=[-8, 12, -3, 4, -4, 4], nb_labels=3,
          xlabel="", ylabel="", zlabel="")
mlab.view(*view)
mlab.savefig('system_exploration/riemann_sheets.png')
