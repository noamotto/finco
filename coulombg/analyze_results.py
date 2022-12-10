# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from finco import load_results

N=10

#%%
logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

cycles = 2
T = halfcycle*2*cycles
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) -0.5*1j*T)

parts = [None] * N
S_Fs = [None] * N
caustics = [None] * N
ts = [None] * N

#%%
results_a = [None] * N
results_g = [None] * N
for n in range(N):
    results_a[n] = load_results(f'res_adaptive_0_15_15_15_t_{cycles}/coulombg_{n}.hdf')
    # results_g[n] = load_results('res_grid_0_5_5_5/coulombg_{}.hdf'.format(n))
    
#%% Stokes treatment
logger.info('Starting treating Stokes')
for i, (result_a, result_g) in enumerate(zip(results_a, results_g)):
    logger.info('Dealing with order {}/{}'.format(i+1, len(results_a)))
    caustics[i] = locate_caustics(result_a, i, T, n_jobs=n_jobs)
    ts[i] = (load_results(f'res_adaptive_0_15_15_15_t_{cycles}/coulombg_{i}.hdf.ct_steps/last_step.hdf').
              get_results(1).t)
    # sig = pd.Series(np.abs(np.angle(ts[i] - 12*np.pi) + np.pi / 2) < np.pi/6, index=ts[i].index)
    S_Fs[i] = eliminate_stokes(result_a, caustics[i])
    # ts[i] = caustic_times(result_a, coulombg_caustic_times_dir, coulombg_caustic_times_dist, n_iters = 180,
    #                       skip = 18, plot_steps=False,
    #                       V = [V_0, V_1, V_2], m = m, gamma_f=1, dt=1, drecord=1, 
    #                       n_jobs=n_jobs, blocksize=2**15, heuristics=[],
    #                       verbose=False)

#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
for i, result in enumerate(results_a):
    logger.info('Reconstructing order {}/{}'.format(i+1, len(results_a)))
    q = result.get_results(1).q
    # mask = (np.abs(np.imag(q)) < 5) & (np.real(q) > -2) & (np.real(q) < 5)
    parts[i] = result.reconstruct_psi(x, 1, S_Fs[i] * (np.imag(ts[i]) > 0), n_jobs=n_jobs, threshold=-1)
    # parts[i] = result.reconstruct_psi(x, 1, S_Fs[i], n_jobs=n_jobs, threshold=-1)
    # parts[i] = result.reconstruct_psi(x, 1, S_Fs[i] * np.sign(np.imag(ts[i])), n_jobs=n_jobs, threshold=-1)
    # parts[i] = result.reconstruct_psi(x, 1, S_Fs[i] * mask, threshold=-1, n_jobs=n_jobs)
    # parts[i] = result.reconstruct_psi(x, 0, n_jobs=n_jobs)

psis = np.cumsum(parts,axis=0)

plt.figure()
plt.plot(x, np.real(y), c='r')
plt.plot(x, np.imag(y), ':', c='r')

for i, psi in enumerate(psis):
    plt.plot(x, np.real(psi), c=plt.cm.winter(i/N))
    plt.plot(x, np.imag(psi), ':', c=plt.cm.winter(i/N))
