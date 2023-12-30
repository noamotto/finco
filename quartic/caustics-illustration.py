# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh

from finco import TimeTrajectory, create_ics, propagate
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, approximate_F
# from utils import tripcolor_complex

plt.rc('font', size=14)

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.INFO)


# System params
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))
    
def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))

def S0_2(q):
    return np.full_like(q, 2j*gamma0)

def V_0(q):
    return a*q**2 + b*q**4
    
def V_1(q):
    return 2*a*q + 4*b*q**3

def V_2(q):
    return 2*a + 12*b*q**2

class QuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics):        
        self.t = np.full_like(ics.q, 0.72)
        
    def t_0(self, tau):
        return self.t * tau
        
    def t_1(self, tau):
        return self.t

class HalfQuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics): 
        self.t0 = ics.t.to_numpy()
        self.t = np.full_like(ics.q, 0.72/2) + self.t0
        
    def t_0(self, tau):
        return (self.t - self.t0) * tau + self.t0
        
    def t_1(self, tau):
        return self.t - self.t0

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass 

#%% Propagate
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 501), np.linspace(-2.5, 2.5, 501))
qs = (X+1j*Y).flatten()

result = propagate(create_ics(qs, S0 = [S0_0, S0_1, S0_2], gamma_f=1), 
                   V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                   time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                   blocksize=1024, n_jobs=5, verbose=True, trajs_path=None)


# trajs = result.get_trajectories(1)
plt.figure('caustics-illustration')
# tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)

#%% Find caustic at bottom-right
deriv = result.get_caustics_map(1)
proj = result.get_projection_map(1)

blobs = separate_to_blobs(deriv, quantile=1e-2)
qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]

caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2], 
                         time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3, n_jobs=3)

#%% Locate and isolate Stokes and anti-Stokes lines

F, F_3 = approximate_F(proj.q0, proj.xi, caustics.iloc[0])
phi0 = np.angle(F_3)

s_lines = []
phis = (np.arange(-4,4) * np.pi - phi0)/3
phis = phis[(phis > -np.pi) & (phis < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - phis[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    s_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).mean())

plt.plot(np.real(s_lines[0]), np.imag(s_lines[0]), 'g')
plt.plot(np.real(s_lines[1]), np.imag(s_lines[1]), 'g')
plt.plot(np.real(s_lines[2]), np.imag(s_lines[2]), 'g')

a_lines = []
phis = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
phis = phis[(phis > -np.pi) & (phis < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - phis[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    # Remove incorrect line in top-right corner that messes up with illustration.
    # mask = ((dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2) & 
    #         ((np.real(proj.q0) < 1) | (np.imag(proj.q0) < 1)))
    
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).mean())

plt.plot(np.real(a_lines[0]), np.imag(a_lines[0]), 'm')
plt.plot(np.real(a_lines[1]), np.imag(a_lines[1]), 'm')
plt.plot(np.real(a_lines[2]), np.imag(a_lines[2]), 'm')

#%% Add Stokes treatment
from matplotlib.colors import ListedColormap

S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
for (i, caustic) in caustics.iterrows():
    S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)

cmap = ListedColormap([[a,a,a] for a in np.linspace(0, 1)])
plt.scatter(np.real(proj.q0), np.imag(proj.q0), c=S_F, cmap=cmap, s=1)
plt.colorbar(label=r"$S(F)$")

#%% Add diverging part
diverging_part = np.real(proj.sigma) > 0

c = [(0,0.7,0.9,int(d)) for d in diverging_part]
tri = Triangulation(np.real(proj.q0), np.imag(proj.q0))
col = TriMesh(tri, facecolors=c, edgecolors='face')
plt.gca().add_collection(col)
plt.gca().autoscale_view()

#%% Image finalization
plt.xlim(-2.5,2.5)
plt.xlabel(r'$\Re q_0$')
plt.ylim(-2.5,2.5)
plt.ylabel(r'$\Im q_0$')
plt.tight_layout()
# plt.savefig('caustics-exploration/caustics-illustration.png')

#%% Iterative stokes line
from joblib import Parallel, delayed
dq = 1e-2

def do_step(result, prev):
    proj = result.get_projection_map(1)
    deriv = result.get_caustics_map(1)
    # print(f'dxi: {(proj.xi.loc[1] - proj.xi.loc[0]).to_numpy()}')
    # print(f'dsigma: {(proj.sigma.loc[1] - proj.sigma.loc[0]).to_numpy()}')
    # print(f'dq0: {(proj.q0.loc[1] - proj.q0.loc[0]).to_numpy()}')
    F1 = deriv.sigma_1.loc[0] - deriv.sigma_1.loc[1] * deriv.xi_1.loc[0] / deriv.xi_1.loc[1]
    phi1 = np.array([np.pi / 2, -np.pi / 2]) - np.angle(F1)
    phi1 = phi1[np.argmax(np.abs(proj.q0.to_numpy()[0]+dq*np.exp(1j*phi1) - prev[0]))]
    dq1 = dq * np.exp(1j *phi1)
    dq2 = (deriv.xi_1.loc[0] / deriv.xi_1.loc[1] * dq1)[1]
    dq1 *= dq / np.max(np.abs([dq1, dq2]))
    dq2 *= dq / np.max(np.abs([dq1, dq2]))
    return create_ics(np.concatenate([proj.q0.loc[0] + dq1,
                                      proj.q0.loc[1] + dq2]), S0 = [S0_0, S0_1, S0_2], gamma_f=1)
    

def iterative_stokes(n, phi, q):
    ics = create_ics([q + dq*np.exp(1j*phi), 
                      q - dq*np.exp(1j*phi)], S0 = [S0_0, S0_1, S0_2], gamma_f=1)
    
    prev = np.array([q, q])
    cur = propagate(ics, 
                    V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                    time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                    blocksize=1024, n_jobs=5, verbose=False, trajs_path=None)
    qs = np.concatenate([[caustic.q], cur.q0.loc[:, 1]])
    
    for i in range(n):
        # print(f'iteration {i+1}')
        next_ics = do_step(cur, prev)
        prev = cur.q0[:,1].to_numpy()
        cur = propagate(next_ics, 
                        V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                        time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                        blocksize=1024, n_jobs=5, verbose=False, trajs_path=None)
        if i % 5 == 0:
            qs = np.concatenate([qs, cur.q0.loc[:, 1]])
    
    return qs

qss = Parallel(verbose=10, n_jobs=3)(delayed(iterative_stokes)(n=550, phi=phis[i], q=caustics.loc[0].q) for i in range(3))
