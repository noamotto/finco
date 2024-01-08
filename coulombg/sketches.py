# -*- coding: utf-8 -*-
"""
File with miscellaneous sketches. Will probably be removed in future commits.
"""

#%% Setup
from coulombg import V, S0, m, CoulombGTimeTrajectory, n_jobs, coulombg_pole, coulombg_diff, halfcycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

from finco import (propagate, 
                   FINCOResults, 
                   load_results, 
                   create_ics, 
                   adaptive_sampling)
from finco.time_traj import TimeTrajectory, CircleTraj, LineTraj
# from finco import NonDivergeingSigmaHeuristic, NonDivergeingMqHeuristic
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, approximate_F, caustic_times
from utils import tripcolor_complex

N=10

#%%
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

x = np.arange(-12, 12, 1e-1)
y = 2**0.5*x*np.exp(-np.abs(x))
parts = [None] * N
S_Fs = [None] * N
caustics = [None] * N
ts = [None] * N

#%%
results_a = [None] * N
results_g = [None] * N
for n in range(N):
    results_a[n] = load_results('res_adaptive_0_15_15_15_2/coulombg_{}.hdf.steps/step_7.hdf'.format(n))
    # results_g[n] = load_results('res_grid_0_5_5_5/coulombg_{}.hdf'.format(n))
    
#%%
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ln = plt.scatter([], [])
def update(frame):
    trajs = results.get_trajs(frame, threshold=-1)
    ln.set_offsets(np.transpose([np.real(trajs.q), np.imag(trajs.q)]))
    ln.set_array(np.log10(np.abs(trajs.pref)+1e-10))
    return ln,

ani = FuncAnimation(fig, update, frames = np.arange(0, 1, 1),
                    interval = 200, blit=True, repeat=False)

#%% Plot poles
X, Y = np.meshgrid(3, np.linspace(-1, -0.8, 20))
qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
qs = qs[(qs != 1j) & (qs != -1j)]


n = np.arange(-10, 11, 1)

ps = S0[1](qs)
Es = ps**2/2/m - 1/qs
sign = np.ones_like(Es)
sign[Es.imag < 0] *= -1

tstars = np.stack([coulombg_pole(qs, ps, n) for n in np.arange(-10, 11, 1)])

plt.scatter(tstars.real, tstars.imag, c=np.tile(qs, (n.size,1)).imag)
# plt.scatter(CoulombGTimeTrajectory.T, 0, c='r')

#%% Trajectory crash test
    
# X, Y = np.meshgrid(np.linspace(1e-10, 5, 302), np.linspace(-4, 4, 302))
# X, Y = np.meshgrid(np.linspace(0, 10, 41), np.linspace(-6, 6, 49))
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
qs = (X+1j*Y)[(np.abs(X + 1j * Y) > 0.01)]
order=2
# qs = qs[(qs != 1j) & (qs != -1j)]
# qs = np.array([(1.6692307692973847-0.8544698544698544j),
#                 (1.6769230769895385-0.8544698544698544j)])
# qs = np.array(-3+3j)

def crash_t(q0, p0):
    return coulombg_pole(q0, p0, np.sign(coulombg_diff(q0, p0).real) * order)

ics = create_ics(qs, S0 = S0, gamma_f=1)
result = propagate(ics,
                   V = V, m = m, gamma_f=1,
                   time_traj = CoulombGTimeTrajectory(n=order, t=crash_t),
                   dt = 1e-4, drecord=1, n_jobs=3)

trajs = result.get_trajectories(1).sort_index()
poles = coulombg_pole(trajs.q0, trajs.p0, 0)
diffs = coulombg_diff(trajs.q0, trajs.p0)
probs = pd.Series(np.imag(poles) - np.imag(diffs) / np.real(diffs)*np.real(poles) < 0)

plt.figure(), plt.scatter(np.real(trajs.q0), np.imag(trajs.q0), c=trajs.q.apply(lambda x: 'b' if np.abs(x) < 1e-1 else 'r'))
plt.figure(), plt.scatter(np.real(trajs.q0), np.imag(trajs.q0), c=probs.apply(lambda x: 'r' if x else 'b'))

#%% Image of the time space
class TestTimeTrajectory(TimeTrajectory):
    def __init__(self, alphas):
        self.alphas = alphas
        
    def init(self, ics):
        q0, p0 = ics.q0.to_numpy(), ics.p0.to_numpy()
        diff = coulombg_diff(q0, p0)
        r_dir = -np.sign(diff)
        self.a = np.array(coulombg_pole(q0, p0, n=0) * (1 - self.alphas) + 
                          coulombg_pole(q0, p0, n=-1) * self.alphas)
        self.r = np.array(-diff * r_dir / 2)

        self.r *= self.alphas * 2

        # Init path
        self.path = [LineTraj(t0=0, t1=0.05, a=0, b=self.a),
                      CircleTraj(t0=0.05, t1=0.1, a=self.a, r=self.r, turns=-0.25, phi0=-np.pi),
                      CircleTraj(t0=0.1, t1=0.5, a=self.a + (1+1j)*self.r, r=self.r, turns=-1, phi0=1*np.pi/2),
                      LineTraj(t0=0.5, t1=0.6, a=self.a + (1+1j)*self.r, 
                               b=self.a + (1+1j + 1 / self.alphas)*self.r),
                      CircleTraj(t0=0.6, t1=1, a=self.a + (1+1j + 1 / self.alphas)*self.r,
                                 r=self.r, turns=-1, phi0=1*np.pi/2)]

        # self.path = [LineTraj(t0=0, t1=0.05, a=0, b=self.a),
        #              LineTraj(t0=0.05, t1=0.1, a=self.a, b=self.a + 0.1 * self.r),
        #              LineTraj(t0=0.1, t1=0.5, a=self.a + 0.1 * self.r, b=self.a + 0.5 * self.r),
        #              LineTraj(t0=0.5, t1=0.6, a=self.a + 0.5 * self.r, b=self.a + 0.6 * self.r),
        #              LineTraj(t0=0.6, t1=1, a=self.a + 0.6 * self.r, b=self.a + 1 * self.r)]
        
        return self

    def t_0(self, tau):
        if tau > 0.6:
            path = 4
        elif tau > 0.5:
            path = 3
        elif tau > 0.1:
            path = 2
        elif tau > 0.05:
            path = 1
        else:
            path = 0
        return self.path[path].t_0(tau)

    def t_1(self, tau):
        if tau > 0.6:
            path = 4
        elif tau > 0.5:
            path = 3
        elif tau > 0.1:
            path = 2
        elif tau > 0.05:
            path = 1
        else:
            path = 0
        return self.path[path].t_1(tau)
    
    def get_discontinuity_times(self):
        return [0.05, 0.1, 0.5, 0.6]

N = 100
q = -0.5-0.5j
qs = np.full(N, q)
alphas = np.linspace(0.99, 0.01, N)

ics = create_ics(qs, S0 = S0, gamma_f=1)
result = propagate(ics, V = V, m = m, gamma_f=1,
                   time_traj=TestTimeTrajectory(alphas=alphas),
                   dt=1e-3, drecord=1/600)

trajs = result.get_trajectories(start=0, end=600)
trajs['E'] = trajs.p**2/2/m + V[0](trajs.q)

time_trajs = TestTimeTrajectory(alphas=alphas).init(ics)
ts = np.stack([time_trajs.t_0(t) for t in np.unique(trajs.index.get_level_values(1))/600]).T.flatten()
tstar = coulombg_pole(np.array([q]), np.array([S0[1](q)]), np.array([-1,0,1]))

plt.figure()
plt.title('q={}'.format(q))
plt.scatter(tstar.real, tstar.imag)
plt.scatter(np.real(ts), np.imag(ts), c=np.real(trajs.p))
plt.colorbar()
plt.scatter(np.real(time_trajs.a), np.imag(time_trajs.a))

#%%
from utils import complex_to_rgb

n=4
result = load_results('res_0_15_12_3_2/coulombg_{}.hdf'.format(n))
shape = (1502, 1002)

# deriv = results[n].get_caustics_map(1).sort_values(by='q0')
# trajs = results[n].get_trajectories(1).sort_values(by='q0')
# proj = results[n].get_projection_map(1).sort_values(by='q0')

deriv = result.get_caustics_map(1).sort_values(by='q0')
trajs = result.get_trajectories(1).sort_values(by='q0')
proj = result.get_projection_map(1).sort_values(by='q0')

grid = np.fliplr(np.reshape(deriv.q0.to_numpy(), shape)).T
xi = np.fliplr(np.reshape(proj.xi.to_numpy(), shape)).T
xi_1 = np.fliplr(np.reshape(deriv.xi_1.to_numpy(), shape)).T
pref = np.fliplr(np.reshape(trajs.pref.to_numpy(), shape)).T
qf = np.real(xi) / 2 / 1
qf[qf < 0] = np.nan
qf[qf > 5] = np.nan
unwrapped = np.abs(pref) * np.exp(1j*np.unwrap(np.angle(pref)))

# plt.figure(), plt.scatter(grid.real, grid.imag, c=complex_to_rgb((xi_1).flatten()))
# plt.figure(), plt.scatter(grid.real, grid.imag, c=complex_to_rgb((xi).flatten(), absmax=1e2))
plt.figure(), plt.scatter(grid.real, grid.imag, c=complex_to_rgb((pref).flatten(), absmax=1e7))
# plt.figure(), plt.pcolormesh(grid.real, grid.imag, qf), plt.colorbar()
# plt.figure(), plt.scatter(grid.real, grid.imag, c=complex_to_rgb((unwrapped).flatten(), absmax=1))

#%%
from coulombg import locate_caustics, eliminate_stokes, coulombg_caustic_times_dir, coulombg_caustic_times_dist
logger = logging.getLogger('finco')
logger.setLevel(logging.DEBUG)

n = 9
caustic = locate_caustics(results_a[n], n, n_jobs=n_jobs)
S_F = eliminate_stokes(results_a[n], caustic)

# ts = caustic_times(results_a[n], coulombg_caustic_times_dir, coulombg_caustic_times_dist, n_iters = 270,
#                     skip = 27, x = x, S_F = S_F, plot_steps=True,
#                     V = V, m = m, gamma_f=1, dt=1, 
#                     n_jobs=n_jobs, blocksize=2**15,
#                     verbose=False) 

#%%
from glob import glob
import re

def mysign(x):
    return (x >= 0).astype(int) * 2 - 1

files = glob(f'res_adaptive_0_15_15_15_2/coulombg_{n}.hdf.ct_steps/step_*')
steps = np.array([int(re.search('.*step_(\d*)\.hdf', i).group(1)) for i in files])

# sort
files = [files[i] for i in np.argsort(steps)]
steps = np.sort(steps)

psis = plt.figure(f'psis {n} 1')
for file, step in zip(files, steps):
    res = load_results(file).get_results(1)
    plt.figure(f'step {step}')
    plt.tripcolor(np.real(res.q0), np.imag(res.q0), np.imag(res.t) >= 0)
    plt.figure(psis)
    plt.plot(x, np.abs(results_a[n].reconstruct_psi(x, 1, (np.imag(res.t) >= 0), n_jobs=n_jobs)),
             c=plt.cm.winter(step / np.max(steps)))
    
plt.draw_all()
    
#%%
import pandas as pd
n=5

proj = results_a[n].get_projection_map(1)
deriv = results_a[n].get_caustics_map(1)
sig = pd.Series(np.abs(np.angle(ts[n] - 12*np.pi) + np.pi / 2) < np.pi/6, index=proj.index)

S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
for (i, caustic) in caustics[n].iterrows():
    logger.debug('handling caustic at {}'.format(caustic.q))
    s_f = calc_factor2(caustic, proj.q0, proj.xi, sig)
    F, _ = approximate_F(proj.q0, proj.xi, caustic)
    r = np.abs(-caustic.xi_2*2/caustic.xi_3)
    # s_f[np.abs(F.v_t) > r] = 1
    S_F *= s_f
    # plt.figure(), plt.tripcolor(np.real(proj.q0), np.imag(proj.q0), S_F), plt.colorbar()
    # plt.scatter(np.real(caustic.q), np.imag(caustic.q))
S_F *= (np.real(proj.sigma) <= 0)
S_F *= (np.abs(deriv.xi_1) <= 100)

#%%
def crash(q,p):
    t = CoulombGTimeTrajectory(n=0).init(create_ics(q,S0,gamma_f=1))
    return coulombg_pole(q,p,t.nfirst + k)
q = np.array([-1 - 4j])
ics = create_ics(q, S0 = S0, gamma_f=1)
plt.figure(fr'$q={q}$')

for k in np.arange(-3,4):
    result = propagate(ics, V = V, m = m, gamma_f = 1,
                            time_traj = CoulombGTimeTrajectory(n=0, t=crash),
                            dt = 1e-3, drecord=1/100,
                            n_jobs = n_jobs, blocksize=2**10,
                            trajs_path=None)
    plt.scatter(np.real(result.t.loc[:,50:]), np.imag(result.t.loc[:,50:]),
                c=np.real(result.p.loc[:, 50:]))
    
#%%
def crash(q,p):
    t = CoulombGTimeTrajectory(n=n).init(create_ics(q,S0,gamma_f=1))
    return t.b + t.u

n=6
X, Y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-6, 6, 300))
qs = (X+1j*Y)[(Y != 1) & (Y != -1)]
ics = create_ics(qs, S0 = S0, gamma_f=1)
result = propagate(ics, V = V, m = m, gamma_f = 1,
                        time_traj = CoulombGTimeTrajectory(n=n, t=crash),
                        dt = 1e-4, drecord=1/100,
                        n_jobs = n_jobs, blocksize=2**10,
                        trajs_path=None)