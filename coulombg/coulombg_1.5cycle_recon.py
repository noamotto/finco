# -*- coding: utf-8 -*-
"""
Analyzes and produces a wavefunction reconstruction of Coulomb ground state
after 1.5 cycles, and produces a figure comparing it to the analytical solution.

Note that while the plot contains the negaitve part of x for legacy reasons, as
the system is only an analytic expansion of the positive part of x the negative
part should be ignored.

To produce the data needed for the reconstruction run the following:
> python ./run_finco_adaptive.py -t 1.5 -o res_adaptive_0_15_15_15_t_1.5 1
> python ./run_finco_adaptive.py -t 1.5 -o res_adaptive_0_15_15_15_t_1.5 2
> python ./run_finco_adaptive.py -t 1.5 -o res_adaptive_0_15_15_15_t_1.5 3
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1.5/coulombg_1_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1.5/coulombg_2_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_1.5/coulombg_3_0.hdf
And extract the contents of the produced .tar.gz files in your working directory.
"""

#%% Setup
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from finco import load_results
from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle

plt.rc('font', size=14)

logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

T = halfcycle*2*1.5
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) + 0.5*1j*T)

try:
    os.mkdir('reconstruction')
except FileExistsError:
    pass

res1 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_1_0.hdf')
res2 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_2_0.hdf')
res3 = load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_3_0.hdf')

#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 1')
caustics1 = locate_caustics(res1, 1, T, n_jobs=n_jobs)
ts1 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_1_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F1 = eliminate_stokes(res1, caustics1)

logger.info('Dealing with order 2')
caustics2 = locate_caustics(res2, 2, T, n_jobs=n_jobs)
ts2 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_2_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F2 = eliminate_stokes(res2, caustics2)

logger.info('Dealing with order 3')
caustics3 = locate_caustics(res3, 3, T, n_jobs=n_jobs)
ts3 = (load_results('res_adaptive_0_15_15_15_t_1.5/coulombg_3_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F3 = eliminate_stokes(res3, caustics3)

#%% Masks
mask1 = np.imag(res1.get_trajectories(1).q0) < -1.6
mask2 = ((np.real(res2.get_projection_map(1).xi) / 2 < 2.5) &
         (np.imag(res2.get_trajectories(1).q0) < 0))
mask3 = ((np.real(res3.get_projection_map(1).xi) / 2 < 1.7) &
         (np.imag(res3.get_trajectories(1).q0) < 0))

#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi1 = res1.reconstruct_psi(x, 1, S_F1 * (np.imag(ts1) > 0) * mask1, n_jobs=n_jobs)
psi2 = res2.reconstruct_psi(x, 1, S_F2 * (np.imag(ts2) > 0) * mask2, n_jobs=n_jobs)
psi3 = res3.reconstruct_psi(x, 1, S_F3 * (np.imag(ts3) > 0) * mask3, n_jobs=n_jobs)

plt.figure('1.5cycles')
plt.title(r'$T=1.5T_g$')
plt.plot(x, np.real(y), 'C0')
plt.plot(x, np.imag(y), ':C0')
plt.plot(x, np.real(psi1+psi2+psi3), 'C1')
plt.plot(x, np.imag(psi1+psi2+psi3), ':C1')

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'],
           fontsize=12)

plt.tight_layout()
plt.savefig('reconstruction/1.5cycles.png')
