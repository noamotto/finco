# -*- coding: utf-8 -*-
"""
Analyzes and produces a wavefunction reconstruction of Coulomb ground state
after 2 cycles, and produces a figure comparing it to the analytical solution.

Note that while the plot contains the negaitve part of x for legacy reasons, as
the system is only an analytic expansion of the positive part of x the negative
part should be ignored.

To produce the data needed for the reconstruction run the following:
> python ./run_finco_adaptive.py -t 2 -o res_adaptive_0_15_15_15_t_2 2
> python ./run_finco_adaptive.py -t 2 -o res_adaptive_0_15_15_15_t_2 3
> python ./run_finco_adaptive.py -t 2 -o res_adaptive_0_15_15_15_t_2 4
> python ./caustic_times.py res_adaptive_0_15_15_15_t_2/coulombg_2_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_2/coulombg_3_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_2/coulombg_4_0.hdf
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

T = halfcycle*2*2
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) + 0.5*1j*T)

try:
    os.mkdir('reconstruction')
except FileExistsError:
    pass

res2 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_2_0.hdf')
res3 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_3_0.hdf')
res4 = load_results('res_adaptive_0_15_15_15_t_2/coulombg_4_0.hdf')

#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 2')
caustics2 = locate_caustics(res2, 2, T, n_jobs=n_jobs)
ts2 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_2_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F2 = eliminate_stokes(res2, caustics2)

logger.info('Dealing with order 3')
caustics3 = locate_caustics(res3, 2, T, n_jobs=n_jobs)
ts3 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_3_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F3 = eliminate_stokes(res3, caustics3)

logger.info('Dealing with order 4')
caustics4 = locate_caustics(res4, 2, T, n_jobs=n_jobs)
ts4 = (load_results('res_adaptive_0_15_15_15_t_2/coulombg_4_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F4 = eliminate_stokes(res4, caustics4)

#%% Masks
mask2 = np.imag(res2.get_trajectories(1).q0) < -1.6
mask3 = ((np.real(res3.get_projection_map(1).xi) / 2 < 2.5) &
         (np.imag(res3.get_trajectories(1).q0) < 0))
mask4 = ((np.real(res4.get_projection_map(1).xi) / 2 < 1.7) &
         (np.imag(res4.get_trajectories(1).q0) < 0))

#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi2 = res2.reconstruct_psi(x, 1, S_F2 * (np.imag(ts2) > 0) * mask2, n_jobs=n_jobs)
psi3 = res3.reconstruct_psi(x, 1, S_F3 * (np.imag(ts3) > 0) * mask3, n_jobs=n_jobs)
psi4 = res4.reconstruct_psi(x, 1, S_F4 * (np.imag(ts4) > 0) * mask4, n_jobs=n_jobs)

plt.figure('2cycles')
plt.title(r'$T=2T_g$')
plt.plot(x, np.real(y), 'C0')
plt.plot(x, np.imag(y), ':C0')
plt.plot(x, np.real(psi2+psi3+psi4), 'C1')
plt.plot(x, np.imag(psi2+psi3+psi4), ':C1')

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'],
           fontsize=12)

plt.tight_layout()
plt.savefig('reconstruction/2cycles.png')
