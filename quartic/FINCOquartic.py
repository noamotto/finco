# -*- coding: utf-8 -*-
"""
Various sketches and miscellaneous pieces of code for propagqation in quaric
potential. Left as reference.

@author: Noam Ottolenghi
"""
#%% Setup

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import propagate, create_ics, adaptive_sampling

#%% Simple propagation

X, Y = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-6, 6, 201))
qs = (X+1j*Y).flatten()
gamma_f = 1
n_steps = 100
# T = 2.
T = 0.72

result = propagate(create_ics(qs, S0 = S0), V = V, m = m, gamma_f=gamma_f,
                   time_traj = QuarticTimeTrajectory(T = T), dt = 1e-4, drecord=1/n_steps,
                   blocksize=2**9, n_jobs=3,
                   trajs_path=f'trajs_{gamma_f}_T_{T}_dt_{T/n_steps}.hdf', verbose=True)

#%% Propagation with adaptive sampling
import logging

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)
n_iters = 10
n_steps = 1
sub_tol = (2e-1,1e3)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 101), np.linspace(-2.5, 2.5, 101))
result, mesh = adaptive_sampling(qs = (X+1j*Y).flatten(), S0 = S0,
                                 n_iters = n_iters, sub_tol = sub_tol, plot_steps=True,
                                 V = V, m = m, gamma_f = 1, blocksize=2**9,
                                 time_traj = QuarticTimeTrajectory(), dt = 1e-4, drecord=1 / n_steps,
                                 n_jobs=3)

#%% Legacy code for plotting trajectory animation
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ln = plt.scatter([], [])
def update(frame):
    trajs = result.get_trajectories(frame, threshold=-1)
    ln.set_offsets(np.transpose([np.real(trajs.q), np.imag(trajs.q)]))
    ln.set_array(np.log10(np.abs(trajs.pref)+1e-10))
    return ln,
ani = FuncAnimation(fig, update, frames = np.arange(0, 10, 5),
                    interval = 200, blit=True, repeat=False)


#%% Functions that Ifound no place for yet. unwrap_Z probably should be moved to a BOMCA related module.
from copy import deepcopy

def extract_params(res, gamma_f=1):
    Z, Pz = res.Mqq + res.Mqp * res.S_20, res.Mpq + res.Mpp * res.S_20
    xi_1 = 2 * gamma_f * Z - 1j * Pz
    return xi_1, Z, Pz

def unwrap_Z(result, step):
    _, Z, _ = extract_params(result.get_results())
    unwrapped = Z.groupby('t_index').transform(lambda x: np.unwrap(np.angle(x)))
    return unwrapped[:,step]

def find_branches2(q0s):
    q0s = np.reshape(np.array(deepcopy(q0s), dtype=object), X.shape)
    q0s[1::2] = q0s[1::2, ::-1]
    q0s = list(q0s.ravel())

    bs = []
    n = np.min([len(n) for n in q0s])

    for i in range(n):
        cur = q0s[0][i]
        q0s[0][i] = np.nan
        br = [i]
        for t in q0s[1:]:
            idx = np.nanargmin(np.abs(cur - t))
            cur = t[idx]
            t[idx] = np.nan
            br.append(idx)
        bs.append(br)

    bs = np.reshape(bs, (-1,) + X.shape)
    bs[:,1::2] = bs[:,1::2,::-1]
    return list(bs.reshape(bs.shape[0], -1))
