# -*- coding: utf-8 -*-
"""
Analyzes and produces a wavefunction reconstruction of Coulomb ground state
after 1 cycle, and produces a figure comparing it to the analytical solution.

Note that while the plot contains the negaitve part of x for legacy reasons, as
the system is only an analytic expansion of the positive part of x the negative
part should be ignored.

To produce the data needed for the reconstruction run the following:
> python ./run_finco_adaptive.py -t 1 -o res_adaptive_0_15_15_15_t_1 0
> python ./run_finco_adaptive.py -t 1 -o res_adaptive_0_15_15_15_t_1 1
> python ./run_finco_adaptive.py -t 1 -o res_adaptive_0_15_15_15_t_1 2
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1/coulombg_0_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1/coulombg_1_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1/coulombg_2_0.hdf
And extract the contents of the produced .tar.gz files in your working directory.
"""

#%% Setup
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle
from finco import load_results

plt.rc('font', size=14)

logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

T = halfcycle*2*1
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) + 0.5*1j*T)

try:
    os.mkdir('reconstruction')
except FileExistsError:
    pass

res0 = load_results('res_adaptive_0_15_15_15_t_1/coulombg_0_0.hdf')
res1 = load_results('res_adaptive_0_15_15_15_t_1/coulombg_1_0.hdf')
res2 = load_results('res_adaptive_0_15_15_15_t_1/coulombg_2_0.hdf')
    
#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 0')
caustics0 = locate_caustics(res0, 0, T, n_jobs=n_jobs)
ts0 = (load_results('res_adaptive_0_15_15_15_t_1/coulombg_0_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F0 = eliminate_stokes(res0, caustics0)

logger.info('Dealing with order 1')
caustics1 = locate_caustics(res1, 1, T, n_jobs=n_jobs)
ts1 = (load_results('res_adaptive_0_15_15_15_t_1/coulombg_1_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F1 = eliminate_stokes(res1, caustics1)

logger.info('Dealing with order 2')
caustics2 = locate_caustics(res2, 2, T, n_jobs=n_jobs)
ts2 = (load_results('res_adaptive_0_15_15_15_t_1/coulombg_2_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F2 = eliminate_stokes(res2, caustics2)

#%% Masks
mask0 = (np.imag(res0.get_trajectories(1).q0) < -1.6)
mask1 = ((np.real(res1.get_projection_map(1).xi) / 2 < 2.5) & 
         (np.imag(res1.get_trajectories(1).q0) < 0))
mask2 = ((np.real(res2.get_projection_map(1).xi) / 2 < 2.1) & 
         (np.imag(res2.get_trajectories(1).q0) < 0))
    
#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi0 = res0.reconstruct_psi(x, 1, S_F0 * (np.imag(ts0) > 0) * mask0, n_jobs=n_jobs)
psi1 = res1.reconstruct_psi(x, 1, S_F1 * (np.imag(ts1) > 0) * mask1, n_jobs=n_jobs)
psi2 = res2.reconstruct_psi(x, 1, S_F2 * (np.imag(ts2) > 0) * mask2, n_jobs=n_jobs)

plt.figure('1cycle')
plt.title(r'$T=T_g$')
plt.plot(x, np.real(y), c=plt.cm.tab10(0))
plt.plot(x, np.imag(y), ':', c=plt.cm.tab10(0))
plt.plot(x, np.real(psi0+psi1+psi2), c=plt.cm.tab10(1))
plt.plot(x, np.imag(psi0+psi1+psi2), ':', c=plt.cm.tab10(1))

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'],
           fontsize=12)

plt.tight_layout()
plt.savefig('reconstruction/1cycle.png')
