# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

from coulombg import V, S0, m, coulombg_pole, coulombg_diff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finco import propagate, create_ics
from finco.time_traj import TimeTrajectory, CircleTraj, LineTraj

class InitialPlaneTimeTrajectory(TimeTrajectory):
    def __init__(self, alphas):
        self.alphas = alphas
        
    def init(self, ics):
        q0, p0 = ics.q0.to_numpy(), ics.p0.to_numpy()
        diff = coulombg_diff(q0, p0)
        poles_dir = -diff/np.abs(diff)
        diff_factor = (1/3-np.abs(self.alphas))*3
        
        t_f = coulombg_pole(q0, p0, n=-1)*(1-self.alphas)/2 + coulombg_pole(q0, p0, n=2)*(self.alphas+1)/2
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
        return self

    def t_0(self, tau):
        if tau < 0.25:
            return self.path[0].t_0(tau)
        elif tau < 0.5:
            return self.path[1].t_0(tau)
        elif tau < 0.75:
            return self.path[2].t_0(tau)
        return self.path[3].t_0(tau)

    def t_1(self, tau):
        if tau < 0.25:
            return self.path[0].t_1(tau)
        elif tau < 0.5:
            return self.path[1].t_1(tau)
        elif tau < 0.75:
            return self.path[2].t_1(tau)
        return self.path[3].t_1(tau)
    
    def get_discontinuity_times(self):
        return [0.25, 0.5, 0.75]

class SecondPlaneTimeTrajectory(TimeTrajectory):
    def __init__(self, alphas):
        self.alphas = alphas
        
    def init(self, ics):
        q0, p0 = ics.q0.to_numpy(), ics.p0.to_numpy()
        diff = coulombg_diff(q0, p0)
        poles_dir = -diff/np.abs(diff)
        
        # First sheet
        t_inner = coulombg_pole(q0, p0, n=0)*(1-self.alphas)/2 + coulombg_pole(q0, p0, n=1)*(self.alphas+1)/2
        a = np.real(poles_dir * np.conj(t_inner)) * poles_dir
        b = t_inner + (t_inner - a) * (self.alphas+1)/2
        c = b - diff * 2 * (self.alphas+1) / 2
        d = t_inner - diff * 2 * (self.alphas+1)/2

        # Second sheet
        e = d - (t_inner - a) * (self.alphas+1)/2
        f = e + 2 * diff * (self.alphas+1)/2
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
        
        return self

    def t_0(self, tau):
        ts = np.array(self.get_discontinuity_times() + [1])
        path = np.flatnonzero(ts >= tau)[0]
        return self.path[path].t_0(tau)

    def t_1(self, tau):
        ts = np.array(self.get_discontinuity_times() + [1])
        path = np.flatnonzero(ts >= tau)[0]
        return self.path[path].t_1(tau)
    
    def get_discontinuity_times(self):
        return [0.05, 0.1, 0.15, 0.2, 0.36, 0.52, 0.68, 0.84]

#%% Initial Plane

q = 1+1j
alphas = np.concatenate([np.linspace(-1, -1/3-1e-2, 20), 
                        np.linspace(-1/3+1e-2, 1/3-1e-2, 40),
                        np.linspace(1/3-1e-2, 1, 20)])
qs = np.full(len(alphas), q)
ics = create_ics(qs, S0 = S0, gamma_f=1)

result1 = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj=InitialPlaneTimeTrajectory(alphas=alphas),
                   dt=1e-3, drecord=1/400, trajs_path=None)

trajs1 = result1.get_trajectories(start=0, end=400)

# ax = plt.figure().add_subplot(projection="3d")
# plt.title('q={}'.format(q))
# ax.scatter(np.real(trajs.t), np.imag(trajs.t), np.real(trajs.p))

plt.figure()
plt.title('q={}'.format(q))
plt.scatter(np.real(trajs1.t), np.imag(trajs1.t), c=np.real(trajs1.p))
plt.colorbar()

#%% Second plane
alphas = np.linspace(-0.975, 0.975, 80)
qs = np.full(len(alphas), q)
ics = create_ics(qs, S0 = S0, gamma_f=1)

result2 = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj=SecondPlaneTimeTrajectory(alphas=alphas),
                   dt=1e-3, drecord=1/500, trajs_path=None)

trajs2 = result2.get_trajectories(start=100, end=500)

# ax.scatter(np.real(trajs.t), np.imag(trajs.t), np.real(trajs.p), color=plt.cm.tab10(1))

plt.figure()
plt.title('q={}'.format(q))
plt.scatter(np.real(trajs2.t), np.imag(trajs2.t), c=np.real(trajs2.p))
plt.colorbar()
