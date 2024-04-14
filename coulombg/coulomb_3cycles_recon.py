# -*- coding: utf-8 -*-
"""
Analyzes and produces a wavefunction reconstruction of Coulomb ground state
after 3 cycles, and produces a figure comparing it to the analytical solution.

Note that while the plot contains the negaitve part of x for legacy reasons, as
the system is only an analytic expansion of the positive part of x the negative
part should be ignored.

To produce the data needed for the reconstruction run the following:
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 4
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 5
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 6
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_4_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_5_0.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_6_0.hdf
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

T = halfcycle*2*3
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) + 0.5*1j*T)

try:
    os.mkdir('reconstruction')
except FileExistsError:
    pass

res4 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_4_0.hdf')
res5 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_5_0.hdf')
res6 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_6_0.hdf')

#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 4')
caustics4 = locate_caustics(res4, 4, T, n_jobs=n_jobs)
ts4 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_4_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F4 = eliminate_stokes(res4, caustics4)

logger.info('Dealing with order 5')
caustics5 = locate_caustics(res5, 5, T, n_jobs=n_jobs)
ts5 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_5_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F5 = eliminate_stokes(res5, caustics5)

logger.info('Dealing with order 6')
caustics6 = locate_caustics(res6, 6, T, n_jobs=n_jobs)
ts6 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_6_0.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F6 = eliminate_stokes(res6, caustics6)

#%% Masks
mask4 = np.imag(res4.get_trajectories(1).q0) < -1.6
mask5 = ((np.real(res5.get_projection_map(1).xi) / 2 < 2.5) &
         (np.imag(res5.get_trajectories(1).q0) < 0))
mask6 = ((np.real(res6.get_projection_map(1).xi) / 2 < 1.5) &
         (np.imag(res6.get_trajectories(1).q0) < 0))

#%% Wavepacket reconstruction
logger.info('Starting wavepacket reconstruction')
psi4 = res4.reconstruct_psi(x, 1, S_F4 * (np.imag(ts4) > 0) * mask4, n_jobs=n_jobs)
psi5 = res5.reconstruct_psi(x, 1, S_F5 * (np.imag(ts5) > 0) * mask5, n_jobs=n_jobs)
psi6 = res6.reconstruct_psi(x, 1, S_F6 * (np.imag(ts6) > 0) * mask6, n_jobs=n_jobs)

plt.figure('3cycles')
plt.title(r'$T=3T_g$')
plt.plot(x, np.real(y), 'C0')
plt.plot(x, np.imag(y), ':C0')
plt.plot(x, np.real(psi4+psi5+psi6), 'C1')
plt.plot(x, np.imag(psi4+psi5+psi6), ':C1')

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', r'FINCO $Re(\psi)$', r'FINCO $Im(\psi)$'],
           fontsize=12)

plt.tight_layout()
plt.savefig('reconstruction/3cycles.png')
