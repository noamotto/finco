# -*- coding: utf-8 -*-
"""
Example of using an interpolator to locate trajectories the end on the real axis,
for BOMCA wavepacket reconstruction.

The code runs a simple propagation of initial conditions in quartic potential,
doing multiple runs to converge as many trajectories as possible. Then it performs
a loop of interpolating initial conditions that end on the real axis and propagating
them, until most points lie up to given threshold from the expexted positions in
real axis.

@author: Noam Ottolenghi
"""
#%% Setup

import numpy as np
import matplotlib.pyplot as plt

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import propagate, create_ics, Mesh, results_from_data
from finco.bomca_interp import BomcaLinearInterpolator
from utils import tripcolor_complex

gamma_f = 1
T = 1

#%% Propagate and refine to converge the most of the trajectories
blocksize = 2**9
n_steps = 100

X, Y = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-6, 6, 201))
qs = (X+1j*Y).flatten()
ics = create_ics(qs, S0 = S0)

result = propagate(ics, V = V, m = m, gamma_f=gamma_f,
                   time_traj = QuarticTimeTrajectory(T = T), dt = 1e-4, drecord=1/n_steps,
                   blocksize=blocksize, n_jobs=3,
                   trajs_path=f'trajs_{gamma_f}_T_{T}_dt_{T/n_steps}.hdf', verbose=True)

while(blocksize >= 1):
    res1 = result.get_results(n_steps)
    dropped = ics.drop(level='t_index', index=np.unique(res1.index.get_level_values('t_index')))
    result2 = propagate(dropped, V = V, m = m, gamma_f=gamma_f,
                       time_traj = QuarticTimeTrajectory(T = T), dt = 1e-4, drecord=1/n_steps,
                       blocksize=blocksize, n_jobs=3,
                       trajs_path=None, verbose=True)
    result.merge(result2)
    blocksize /= 2**3

#%% Iterate for root-search of points on the real axis
n = 5
qsamples = np.linspace(-5, 5, 500)
tol = 1e-5

for i in range(n):
    res1 = result.get_results(n_steps)
    deriv = result.get_caustics_map(n_steps)
        
    S = res1.S + 0.5j * np.log(deriv.Z)
    plt.figure(), tripcolor_complex(np.real(res1.q0), np.imag(res1.q0), np.exp(1j*S), absmax=1e7)
    plt.xlabel(r'$\Re q_0$'), plt.ylabel(r'$\Im q_0$')
    
    bomca = BomcaLinearInterpolator(result, n_steps)
    mask = np.std(res1.q.to_numpy().take(bomca.simplices), axis=1) < 1
    q0s = bomca(qsamples, res1.q0, mask, ablocks=5, bblocks=5)
    plt.scatter(np.concatenate(q0s).real, np.concatenate(q0s).imag, c='r', s=2)
    
    expected = np.concatenate([np.full_like(q0, x) for q0, x in zip(q0s, qsamples)])
    mesh = Mesh(res1, adaptive=True)
    ics = create_ics(np.concatenate(q0s), S0)
    _, ics = mesh.add_points(ics)
    
    result2 = propagate(ics, V = V, m = m, gamma_f=gamma_f,
                       time_traj = QuarticTimeTrajectory(T = T), dt = 1e-4, drecord=1/n_steps,
                       blocksize=2**9, n_jobs=3, trajs_path=None, verbose=True)
    res2 = result2.get_results()
    got = res2.loc[:, n_steps, :].q
    diff = np.abs(expected - got)
    
    tomerge = res2.loc[np.unique(res2.index.get_level_values('t_index'))[diff > tol]]
    result3 = results_from_data(tomerge)
    result.merge(result3)
    
    print(f'Iteration {i+1}/{n} summary')
    print(f'Number of considered points: {len(expected)}')
    print(f'Number of added points: {len(tomerge) // (n_steps + 1)}')
    print('----------------------------------')
    print('Difference from root-searched value:')
    print(f'mean: {np.mean(diff):-g}, std: {np.std(diff):-g}')
    print(f'min: {np.min(diff):-g}, max: {np.max(diff):-g}')
    print('##################################')
