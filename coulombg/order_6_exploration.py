# -*- coding: utf-8 -*-
"""
Produces the maps of xi_1 and prefactor for "order" n=6 in propagation of 3 cycles.

To produce the data needed for the exploration run the following:
> python ./run_finco_adaptive.py -t 3 -o res_adaptive_0_15_15_15_t_3 6
> python ./caustic_times.py res_adaptive_0_15_15_15_t_3/coulombg_6.hdf
"""

#%% Setup
import os

from coulombg import locate_caustics, eliminate_stokes, n_jobs, halfcycle, CoulombGTimeTrajectory

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

from finco import load_results
from utils import tripcolor_complex, complex_to_rgb

plt.rc('font', size=14)

logging.getLogger('finco').setLevel(logging.INFO)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

T = halfcycle*2*3
x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x) -0.5*1j*T)

try:
    os.mkdir('order_6')
except FileExistsError:
    pass

res = load_results('res_adaptive_0_15_15_15_t_3/coulombg_6.hdf')
trajs = res.get_trajectories(1)
deriv = res.get_caustics_map(1)
    
#%% Stokes treatment
caustics = locate_caustics(res, 6, T, n_jobs=n_jobs)
ts = (load_results('res_adaptive_0_15_15_15_t_3/coulombg_6.hdf.ct_steps/last_step.hdf').
          get_results(1).t)
S_F = eliminate_stokes(res, caustics)
    
#%% Produce maps
fig, (xi_1, prefactor) = plt.subplots(1, 2, num='pref_xi_1', figsize=(9.6, 4.8))

plt.sca(prefactor), tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0),
                                      trajs.pref, absmax=1e7)
prefactor.set_xlim(0,15)
prefactor.set_ylim(-15,15)
prefactor.set_xlabel(r'$\Re q_0$')
prefactor.set_ylabel(r'$\Im q_0$')
prefactor.set_title(r'$\phi$')

plt.sca(xi_1), tripcolor_complex(np.real(deriv.q0), np.imag(deriv.q0), 
                                 deriv.xi_1, absmax=1e2)
xi_1.set_xlim(0,15)
xi_1.set_ylim(-15,15)
xi_1.set_xlabel(r'$\Re q_0$')
xi_1.set_ylabel(r'$\Im q_0$')
xi_1.set_title(r'$d\xi_f/dq_0$')

fig.tight_layout()
fig.savefig('order_6/pref_xi_1.png')

#%% Produce segmented map
stokes_mask = S_F * (np.imag(ts) > 0)
tail_mask = S_F * (np.imag(ts) > 0) * (np.real(trajs.q0) > 4) * (np.abs(trajs.pref) > 1e-10)

stokes_cmap = ListedColormap([[1,0,0,a] for a in np.linspace(.2, 0)])
tail_cmap = ListedColormap([[0,1,0,a] for a in np.linspace(0, .1)])

plt.figure('pref_segmented')
tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)
plt.tricontourf(np.real(trajs.q0), np.imag(trajs.q0), stokes_mask, cmap=stokes_cmap)
plt.tricontourf(np.real(trajs.q0), np.imag(trajs.q0), tail_mask, cmap=tail_cmap)
plt.xlabel(r'$\Re q_0$')
plt.ylabel(r'$\Im q_0$')
plt.tight_layout()
plt.savefig('order_6/pref_segmented.png')

#%% Final position figures
stokes_c = complex_to_rgb(trajs.pref, absmax=1e7)
stokes_c[:,3] = stokes_mask
tail_c = complex_to_rgb(trajs.pref, absmax=1e7)
tail_c[:,3] = tail_mask

fig, (no_filter, stokes_filter, tail_filter) = plt.subplots(1, 3, num='pref_q',
                                                            figsize=(14.4, 4.8))
no_filter.scatter(np.real(trajs.q), np.imag(trajs.q), c=complex_to_rgb(trajs.pref, absmax=1e7))
no_filter.set_xlim(-5, 110)
no_filter.set_ylim(-10, 10)
no_filter.set_xlabel(r'$\Re q$')
no_filter.set_ylabel(r'$\Im q$')
no_filter.set_title('(a)')

stokes_filter.scatter(np.real(trajs.q), np.imag(trajs.q), c = stokes_c)
stokes_filter.set_xlim(-5, 110)
stokes_filter.set_ylim(-10, 10)
stokes_filter.set_xlabel(r'$\Re q$')
stokes_filter.set_ylabel(r'$\Im q$')
stokes_filter.set_title('(b)')

tail_filter.scatter(np.real(trajs.q), np.imag(trajs.q), c = tail_c)
tail_filter.set_xlim(-5, 110)
tail_filter.set_ylim(-10, 10)
tail_filter.set_xlabel(r'$\Re q$')
tail_filter.set_ylabel(r'$\Im q$')
tail_filter.set_title('(c)')

fig.tight_layout()
fig.savefig('order_6/pref_q.png')

#%% End of circumnavigation time figures
tend = CoulombGTimeTrajectory(6, t=T).init(res.get_results(0,1)).b

fig, (no_filter, stokes_filter, tail_filter) = plt.subplots(1, 3, num='pref_tend',
                                                            figsize=(14.4, 4.8))
no_filter.scatter(np.real(tend), np.imag(tend), c=complex_to_rgb(trajs.pref, absmax=1e7))
no_filter.set_xlim(-150, 150)
no_filter.set_ylim(-150, 150)
no_filter.set_xlabel(r'$\Re t_e$', fontsize=16)
no_filter.set_ylabel(r'$\Im t_e$', fontsize=16)
no_filter.set_title('(a)')

stokes_filter.scatter(np.real(tend), np.imag(tend), c = stokes_c)
stokes_filter.set_xlim(-25, 75)
stokes_filter.set_ylim(-40, 20)
stokes_filter.set_xlabel(r'$\Re t_e$', fontsize=16)
stokes_filter.set_ylabel(r'$\Im t_e$', fontsize=16)
stokes_filter.set_title('(b)')

tail_filter.scatter(np.real(tend), np.imag(tend), c = tail_c)
tail_filter.set_xlim(-25, 75)
tail_filter.set_ylim(-40, 20)
tail_filter.set_xlabel(r'$\Re t_e$', fontsize=16)
tail_filter.set_ylabel(r'$\Im t_e$', fontsize=16)
tail_filter.set_title('(c)')

fig.tight_layout()
fig.savefig('order_6/pref_tend.png')

#%% Reconstruction
psi_stokes = res.reconstruct_psi(x, 1, stokes_mask, n_jobs=n_jobs)
psi_tail = res.reconstruct_psi(x, 1, tail_mask, n_jobs=n_jobs)

plt.figure('reconstruction')
plt.plot(x, np.real(y), c=plt.cm.tab10(0))
plt.plot(x, np.imag(y), ':', c=plt.cm.tab10(0))
plt.plot(x, np.real(psi_stokes), c=plt.cm.tab10(1))
plt.plot(x, np.imag(psi_stokes), ':', c=plt.cm.tab10(1))
plt.plot(x, np.real(psi_tail), c=plt.cm.tab10(2))
plt.plot(x, np.imag(psi_tail), ':', c=plt.cm.tab10(2))

plt.xlabel(r'$x$')
plt.legend([r'QM $Re(\psi)$', r'QM $Im(\psi)$', 
            r'FINCO+Stokes treatment $Re(\psi)$', r'FINCO+Stokes treatment $Im(\psi)$',
            r'FINCO+tail only $Re(\psi)$', r'FINCO+tail only $Im(\psi)$'], fontsize=10)
plt.tight_layout()
plt.savefig('order_6/reconstruction.png')
