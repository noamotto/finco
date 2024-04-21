#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the behavior of a pole jump in Coulomb potential with no external field.

The script calculates the trajectory in time for a circumanvigation of the origin
in position space that contains the zero at :math:`-\\frac{e}{E0}`, which results
in a pole jump. It then plots the circumnavigation part in position, momentum and
time as a scatter plot for a trajectory with and without pole jump. To make the
propagation direction visible the markers for the trajectory gradually change
color from cyan to magenta. In addition, the analytical position of the momentum
poles is plotted in red, and the position of the momentum zeros is plotted in green.

@author: Noam Ottolenghi
"""
#%% Setup
import numpy as np
import matplotlib.pyplot as plt

from finco import create_ics
from finco.time_traj import SequentialTraj, LineTraj, CircleTraj
from finco.coord2time import Space2TimeTraj
from coulomb import m, q_e
from coulombg import coulombg_pole, V as V_nofield

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

ics = create_ics(q0, S0)
E0 = np.array(ics.p**2/2/m + V_nofield[0](ics.q,0))
r = np.abs(q_e / E0)

s = np.linspace(0.3,1,100)
tstars = coulombg_pole(q0, ics.p.to_numpy(), n=np.arange(0,2))

#%% Run
alpha = 1.05 # Set to a number higher than 1
nojump = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r/alpha), V=V_nofield, m=m, max_step=1e-5).init(ics)
jump = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r*alpha), V=V_nofield, m=m, max_step=1e-5).init(ics)

#%% Plot graphs with no jump
# Plotting parameters were chosen based on the propagation result
ts = np.array([nojump.t_0(_s) for _s in s])
qs = np.array([nojump.q_traj.t_0(_s) for _s in s])
ps = np.array([nojump.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='nojump', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0,0, c='r')
q.scatter(np.real(-q_e / E0), np.imag(-q_e / E0), c='g')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_ylabel(r'$\Im q$')

vals = p.scatter(ps.real, ps.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.scatter(tstars.real, tstars.imag, c='r')
tau.scatter(np.real((tstars[0] +  tstars[1]) / 2),
            np.imag((tstars[0] +  tstars[1]) / 2), c='g')
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_xlim(-2, 3.5)
tau.set_ylabel(r'$\Im t$')
tau.set_ylim(-0.3, 5.1)

plt.tight_layout()

#%% Plot graphs with jump
# Plotting parameters were chosen based on the propagation result
ts = np.array([jump.t_0(_s) for _s in s])
qs = np.array([jump.q_traj.t_0(_s) for _s in s])
ps = np.array([jump.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='jump', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0,0, c='r')
q.scatter(np.real(-q_e / E0), np.imag(-q_e / E0), c='g')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_ylabel(r'$\Im q$')

vals = p.scatter(ps.real, ps.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.scatter(tstars.real, tstars.imag, c='r')
tau.scatter(np.real((tstars[0] +  tstars[1]) / 2),
            np.imag((tstars[0] +  tstars[1]) / 2), c='g')
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_xlim(-2, 3.5)
tau.set_ylabel(r'$\Im t$')
tau.set_ylim(-0.3, 5.1)

plt.tight_layout()
