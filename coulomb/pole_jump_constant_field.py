#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the behavior of a pole jump in Coulomb potential with constant external field.

The script calculates the trajectory in time for a circumanvigation of the origin
in position space that contains none, one or two of the momentum zeros at

.. math:: q_{\\pm} = \\frac{-E\\pm\\sqrt{E^2 - 4eA_0}}{2A_0}

which results in the different behaviors.

It then plots the circumnavigation part in position, momentum and time as a scatter
plot for a trajectory in each behavior. To make the propagation direction visible
the markers for the trajectory gradually change color from cyan to magenta. In
addition, the analytical position of the momentum poles is plotted in red, and
the position of the momentum zeros is plotted in green.

@author: Noam Ottolenghi
"""
#%% Setup

import numpy as np
import matplotlib.pyplot as plt

from coulomb import m

from finco import create_ics
from finco.time_traj import SequentialTraj, LineTraj, CircleTraj
from finco.coord2time import Space2TimeTraj

q_e = 1
A0 = 1e0
def S0_mock(_):
    """
    Mock S0 funcion to manually set initial momenta

    Parameters
    ----------
    q0 : ArrayLike
        Initial positions.
    """
    return p0

S0 = [S0_mock] * 3

def V_0(q, t):
    return -q_e / q - A0 * q

def V_1(q, t):
    return q_e / q**2 - A0

def V_2(q, t):
    return -2 * q_e / q**3

V_constfield = [V_0, V_1, V_2]

class ZeroLocatorTraj(SequentialTraj):
    def __init__(self, r, qstar):
        super().__init__(t0=0, t1=1)
        self.r = r
        self.qstar = qstar

    def init(self, ics):
        q0 = ics.q.to_numpy()
        r = np.abs(self.r / q0) * q0
        self.path = [LineTraj(t0=0, t1=0.3, a=q0, b=r),
                     CircleTraj(t0=0.3, t1=0.6, a=r, r=r, turns=1, phi0=0),
                     LineTraj(t0=0.6, t1=1, a=r, b=self.qstar)]
        self.discont_times = [0.3, 0.6]

        return self

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


q0 = np.array([1-1j])
p0 = np.array([1+1j])
ics = create_ics(q0, S0)

E0 = np.array(ics.p**2/2/m + V_constfield[0](ics.q,0))
qstars = np.concatenate([(-E0+(E0**2-4*A0*q_e)**0.5)/2/A0,
                         (-E0-(E0**2-4*A0*q_e)**0.5)/2/A0])
qstars = qstars[np.argsort(np.abs(qstars))]
rs = np.abs(qstars) * q0 / np.abs(q0)
r_nojump = rs[0] * 0.9
r_jump = 0.99 * rs[0] + 0.01 * rs[1]
r_field = rs[1] / 0.95

s = np.linspace(0.3,1,100)

#%% Run

# Locate relevant momentum poles
pole0 = Space2TimeTraj(t0=0, t1=1, q_traj=ZeroLocatorTraj(r_nojump, 1e-3),
                       V=V_constfield, m=m, max_step=1e-4).init(ics)
pole1 = Space2TimeTraj(t0=0, t1=1, q_traj=ZeroLocatorTraj(r_jump, 1e-3),
                       V=V_constfield, m=m, max_step=1e-4).init(ics)

poles = np.array([pole0.t_0(1), pole1.t_0(1)])

# Locate relevant momentum zeros
zero1 = Space2TimeTraj(t0=0, t1=1, q_traj=LineTraj(t0=0, t1=1, a=q0, b=qstars[0]),
                       V=V_constfield, m=m, max_step=1e-4).init(ics)
zero2 = Space2TimeTraj(t0=0, t1=1, q_traj=ZeroLocatorTraj(r_field, qstars[1]),
                       V=V_constfield, m=m, max_step=1e-4).init(ics)

zeros = np.array([zero1.t_0(1), zero2.t_0(1)])

nojump = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r_nojump), V=V_constfield,
                        m=m, max_step=1e-4).init(ics)
jump = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r_jump), V=V_constfield,
                      m=m, max_step=1e-4).init(ics)
field = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r_field), V=V_constfield,
                       m=m, max_step=1e-4).init(ics)

#%% Plot no jump
ts = np.array([nojump.t_0(_s) for _s in s])
qs = np.array([nojump.q_traj.t_0(_s) for _s in s])
ps = np.array([nojump.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='const_nojump', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0, 0, c='r')
q.scatter(np.real(qstars), np.imag(qstars), c='g')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_xlim(-0.8, 1.4)
q.set_ylabel(r'$\Im q$')
q.set_ylim(-2, 0.6)

vals = p.scatter(ps.real, ps.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.scatter(np.real(poles), np.imag(poles), c='r')
tau.scatter(np.real([zero1.t_0(1), zero2.t_0(1)]),
            np.imag([zero1.t_0(1), zero2.t_0(1)]), c='g')
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_xlim(-0.7, 0.2)
tau.set_ylabel(r'$\Im t$')
tau.set_ylim(-0.6, 1.55)

plt.tight_layout()

#%% Plot pole jump
ts = np.array([jump.t_0(_s) for _s in s])
qs = np.array([jump.q_traj.t_0(_s) for _s in s])
ps = np.array([jump.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='const_jump', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0, 0, c='r')
q.scatter(np.real(qstars), np.imag(qstars), c='g')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_xlim(-0.8, 1.4)
q.set_ylabel(r'$\Im q$')
q.set_ylim(-2, 0.6)

vals = p.scatter(ps.real, ps.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.scatter(np.real(poles), np.imag(poles), c='r')
tau.scatter(np.real([zero1.t_0(1), zero2.t_0(1)]),
            np.imag([zero1.t_0(1), zero2.t_0(1)]), c='g')
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_xlim(-0.7, 0.2)
tau.set_ylabel(r'$\Im t$')
tau.set_ylim(-0.6, 1.55)

plt.tight_layout()

#%% Plot pole "field" jump
ts = np.array([field.t_0(_s) for _s in s])
qs = np.array([field.q_traj.t_0(_s) for _s in s])
ps = np.array([field.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='const_field', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, s=5, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0, 0, c='r')
q.scatter(np.real(qstars), np.imag(qstars), c='g')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_xlim(-2.6, 2.6)
q.set_ylabel(r'$\Im q$')
q.set_ylim(-2.6, 2.6)

vals = p.scatter(ps.real, ps.imag, c=s, s=5, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, s=5, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.scatter(np.real(poles), np.imag(poles), c='r')
tau.scatter(np.real([zero1.t_0(1), zero2.t_0(1)]),
            np.imag([zero1.t_0(1), zero2.t_0(1)]), c='g')
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_xlim(-1.65, 1.3)
tau.set_ylabel(r'$\Im t$')
tau.set_ylim(-1, 2.2)

plt.tight_layout()
