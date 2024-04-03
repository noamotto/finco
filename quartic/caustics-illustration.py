# -*- coding: utf-8 -*-
"""
Illustration of the behavior of caustics and Stokes treatment we do to eliminate
unphysical regions.

The example propagates a Gaussian in quartic potential, the looks for the caustics
and applies Stokes treatment. It then plots the applied factor to each trajectory
(transparent to black color), the Stokes lines (green) and anti-Stokes lines (magenta)
for the bottom-left caustic, and the diverging parts in blue.

@author: Noam Ottolenghi
"""
#%% Setup
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import create_ics, propagate
from finco.stokes import separate_to_blobs, calc_factor2, approximate_F, find_caustics
# Uncomment to also draw the trajectory prefactor. Results in less clean image..
# from utils import tripcolor_complex

plt.rc('font', size=14)

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.INFO)

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

#%% Propagate
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 501), np.linspace(-2.5, 2.5, 501))
qs = (X+1j*Y).flatten()

result = propagate(create_ics(qs, S0 = S0),
                   V = V, m = m, gamma_f=1,
                   time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                   blocksize=1024, n_jobs=5, verbose=True, trajs_path=None)


plt.figure('caustics-illustration')
# Uncomment to also draw the trajectory prefactor. Results in less clean image..
# trajs = result.get_trajectories(1)
# tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)

#%% Find caustic at bottom-right
deriv = result.get_caustics_map(1)
proj = result.get_projection_map(1)

blobs = separate_to_blobs(deriv, quantile=1e-2)
qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]

caustics = find_caustics(qs, V = V, m = m, S0 = S0,
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
    # Remove incorrect line in top-right corner that messes up with illustration.
    mask = ((dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2) &
            ((np.real(proj.q0) < 1) | (np.imag(proj.q0) < 1)))

    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).mean())

plt.plot(np.real(a_lines[0]), np.imag(a_lines[0]), 'm')
plt.plot(np.real(a_lines[1]), np.imag(a_lines[1]), 'm')
plt.plot(np.real(a_lines[2]), np.imag(a_lines[2]), 'm')

#%% Add Stokes treatment
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
