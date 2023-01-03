# -*- coding: utf-8 -*-
"""
Produces prefactor maps for orders n=4,5,6 for comparison of contributing orders.

To produce the data needed for this analysis run the following:
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 4
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 5
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 6
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_4.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_5.hdf
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_6.hdf
"""

#%% Setup
import os

from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

from finco import load_results
from utils import tripcolor_complex

try:
    os.mkdir('orders_comparison')
except FileExistsError:
    pass

logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

T = halfcycle*2*3
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) -0.5*1j*T)

res4 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_4.hdf')
res5 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_5.hdf')
res6 = load_results('res_adaptive_0_15_15_15_t_3/coulombg_6.hdf')
    
#%% Stokes treatment
logger.info('Starting treating Stokes')

logger.info('Dealing with order 4')
caustics4 = locate_caustics(res4, 4, T, n_jobs=n_jobs)
ts4 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_4.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F4 = eliminate_stokes(res4, caustics4)

logger.info('Dealing with order 5')
caustics5 = locate_caustics(res5, 5, T, n_jobs=n_jobs)
ts5 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_5.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F5 = eliminate_stokes(res5, caustics5)

logger.info('Dealing with order 6')
caustics6 = locate_caustics(res6, 6, T, n_jobs=n_jobs)
ts6 = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_6.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F6 = eliminate_stokes(res6, caustics6)

#%% Masks
mask4 = (np.imag(res4.get_trajectories(1).q0) < -1.6)
mask5 = ((np.real(res5.get_projection_map(1).xi) / 2 < 2.5) & 
         (np.imag(res5.get_trajectories(1).q0) < 0))
mask6 = ((np.real(res6.get_projection_map(1).xi) / 2 < 1.5) & 
         (np.imag(res6.get_trajectories(1).q0) < 0))
    
#%% Prefactor maps
stokes_cmap = ListedColormap([[1,0,0,a] for a in np.linspace(.2, 0)])

trajs4 = res4.get_trajectories(1)
trajs5 = res5.get_trajectories(1)
trajs6 = res6.get_trajectories(1)

fig, (pref4, pref5, pref6) = plt.subplots(1, 3, num='prefs_segmented', figsize=(14.4, 4.8))

plt.sca(pref4)
tripcolor_complex(np.real(trajs4.q0), np.imag(trajs4.q0), trajs4.pref, absmax=1e7)
pref4.tricontourf(np.real(trajs4.q0), np.imag(trajs4.q0), S_F4 * (np.imag(ts4) > 0), cmap=stokes_cmap)
pref4.set_title(r"$n=4$")
pref4.set_xlabel(r"$\Re q_0$")
pref4.set_ylabel(r"$\Im q_0$")

plt.sca(pref5)
tripcolor_complex(np.real(trajs5.q0), np.imag(trajs5.q0), trajs5.pref, absmax=1e7)
pref5.tricontourf(np.real(trajs5.q0), np.imag(trajs5.q0), S_F5 * (np.imag(ts5) > 0), cmap=stokes_cmap)
pref5.set_title(r"$n=5$")
pref5.set_xlabel(r"$\Re q_0$")
pref5.set_ylabel(r"$\Im q_0$")

plt.sca(pref6)
tripcolor_complex(np.real(trajs6.q0), np.imag(trajs6.q0), trajs6.pref, absmax=1e7)
pref6.tricontourf(np.real(trajs6.q0), np.imag(trajs6.q0), S_F6 * (np.imag(ts6) > 0), cmap=stokes_cmap)
pref6.set_title(r"$n=6$")
pref6.set_xlabel(r"$\Re q_0$")
pref6.set_ylabel(r"$\Im q_0$")

fig.tight_layout()
fig.savefig('orders_comparison/prefs_segmented.png')

#%% Prefactor maps with mask
stokes_cmap = ListedColormap([[1,0,0,a] for a in np.linspace(.2, 0)])
mask_cmap = ListedColormap([[0,1,0,a] for a in np.linspace(0, .2)])

trajs4 = res4.get_trajectories(1)
trajs5 = res5.get_trajectories(1)
trajs6 = res6.get_trajectories(1)

fig, (pref4, pref5, pref6) = plt.subplots(1, 3, num='prefs_segmented_masked', figsize=(14.4, 4.8))

plt.sca(pref4)
tripcolor_complex(np.real(trajs4.q0), np.imag(trajs4.q0), trajs4.pref, absmax=1e7)
pref4.tricontourf(np.real(trajs4.q0), np.imag(trajs4.q0), S_F4 * (np.imag(ts4) > 0), cmap=stokes_cmap)
pref4.tricontourf(np.real(trajs4.q0), np.imag(trajs4.q0), 
                  (1 - mask4) * (S_F4 * (np.imag(ts4) > 0)), cmap=mask_cmap)
pref4.set_title(r"$n=4$")
pref4.set_xlabel(r"$\Re q_0$")
pref4.set_ylabel(r"$\Im q_0$")

plt.sca(pref5)
tripcolor_complex(np.real(trajs5.q0), np.imag(trajs5.q0), trajs5.pref, absmax=1e7)
pref5.tricontourf(np.real(trajs5.q0), np.imag(trajs5.q0), S_F5 * (np.imag(ts5) > 0), cmap=stokes_cmap)
pref5.tricontourf(np.real(trajs5.q0), np.imag(trajs5.q0), 
                  (1 - mask5) * (S_F5 * (np.imag(ts5) > 0)), cmap=mask_cmap)
pref5.set_title(r"$n=5$")
pref5.set_xlabel(r"$\Re q_0$")
pref5.set_ylabel(r"$\Im q_0$")

plt.sca(pref6)
tripcolor_complex(np.real(trajs6.q0), np.imag(trajs6.q0), trajs6.pref, absmax=1e7)
pref6.tricontourf(np.real(trajs6.q0), np.imag(trajs6.q0), S_F6 * (np.imag(ts6) > 0), cmap=stokes_cmap)
pref6.tricontourf(np.real(trajs6.q0), np.imag(trajs6.q0), 
                  (1 - mask6) * (S_F6 * (np.imag(ts6) > 0)), cmap=mask_cmap)
pref6.set_title(r"$n=6$")
pref6.set_xlabel(r"$\Re q_0$")
pref6.set_ylabel(r"$\Im q_0$")

fig.tight_layout()
fig.savefig('orders_comparison/prefs_segmented_masked.png')