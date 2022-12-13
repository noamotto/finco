# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup

from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle

import numpy as np
import matplotlib.pyplot as plt
import logging

from finco import load_results

logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

T = halfcycle*2*1.5
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) -0.5*1j*T)

res1 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_1.hdf')
res2 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_2.hdf')
res3 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_3.hdf')
    
#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 1')
caustics1 = locate_caustics(res1, 1, T, n_jobs=n_jobs)
ts1 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_1.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F1 = eliminate_stokes(res1, caustics1)

logger.info('Dealing with order 2')
caustics2 = locate_caustics(res2, 2, T, n_jobs=n_jobs)
ts2 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_2.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F2 = eliminate_stokes(res2, caustics2)

logger.info('Dealing with order 3')
caustics3 = locate_caustics(res3, 3, T, n_jobs=n_jobs)
ts3 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_3.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F3 = eliminate_stokes(res3, caustics3)

#%% Masks
mask1 = (np.imag(res1.get_trajectories(1).q0) < -1.6)
mask2 = ((np.real(res2.get_projection_map(1).xi) / 2 < 2.5) & 
         (np.imag(res2.get_trajectories(1).q0) < 0))
mask3 = ((np.real(res3.get_projection_map(1).xi) / 2 < 1.7) & 
         (np.imag(res3.get_trajectories(1).q0) < 0))
    
#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi1 = res1.reconstruct_psi(x, 1, S_F1 * (np.imag(ts1) > 0) * mask1, n_jobs=n_jobs)
psi2 = res2.reconstruct_psi(x, 1, S_F2 * (np.imag(ts2) > 0) * mask2, n_jobs=n_jobs)
psi3 = res3.reconstruct_psi(x, 1, S_F3 * (np.imag(ts3) > 0) * mask3, n_jobs=n_jobs)

plt.figure('1.5cycle')
plt.plot(x, np.real(y), c=plt.cm.tab10(0))
plt.plot(x, np.imag(y), ':', c=plt.cm.tab10(0))
plt.plot(x, np.real(psi1+psi2+psi3), c=plt.cm.tab10(1))
plt.plot(x, np.imag(psi1+psi2+psi3), ':', c=plt.cm.tab10(1))

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'])
