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

T = halfcycle*2*2
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) -0.5*1j*T)

res2 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_2.hdf')
res3 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_3.hdf')
res4 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_4.hdf')
    
#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 2')
caustics2 = locate_caustics(res2, 2, T, n_jobs=n_jobs)
ts2 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_2.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F2 = eliminate_stokes(res2, caustics2)

logger.info('Dealing with order 3')
caustics3 = locate_caustics(res3, 2, T, n_jobs=n_jobs)
ts3 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_3.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F3 = eliminate_stokes(res3, caustics3)

logger.info('Dealing with order 4')
caustics4 = locate_caustics(res4, 2, T, n_jobs=n_jobs)
ts4 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_4.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F4 = eliminate_stokes(res4, caustics4)

#%% Masks
mask2 = (np.imag(res2.get_trajectories(1).q0) < -1.6)
mask3 = ((np.real(res3.get_projection_map(1).xi) / 2 < 2.5) & 
         (np.imag(res3.get_trajectories(1).q0) < 0))
mask4 = ((np.real(res4.get_projection_map(1).xi) / 2 < 1.7) & 
         (np.imag(res4.get_trajectories(1).q0) < 0))
    
#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi2 = res2.reconstruct_psi(x, 1, S_F2 * (np.imag(ts2) > 0) * mask2, n_jobs=n_jobs)
psi3 = res3.reconstruct_psi(x, 1, S_F3 * (np.imag(ts3) > 0) * mask3, n_jobs=n_jobs)
psi4 = res4.reconstruct_psi(x, 1, S_F4 * (np.imag(ts4) > 0) * mask4, n_jobs=n_jobs)

plt.figure('2cycle')
# plt.clf()
plt.plot(x, np.real(y), c=plt.cm.tab10(0))
plt.plot(x, np.imag(y), ':', c=plt.cm.tab10(0))
plt.plot(x, np.real(psi2+psi3+psi4), c=plt.cm.tab10(1))
plt.plot(x, np.imag(psi2+psi3+psi4), ':', c=plt.cm.tab10(1))

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'])
