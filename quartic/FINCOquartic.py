# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from finco import propagate, continue_propagation, create_ics, TimeTrajectory, load_results, adaptive_sampling
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, caustic_times
from utils import tripcolor_complex
 

# System params
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))
    
def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))

def S0_2(q):
    return np.full_like(q, 2j*gamma0)

def V_0(q):
    return a*q**2 + b*q**4
    
def V_1(q):
    return 2*a*q + 4*b*q**3

def V_2(q):
    return 2*a + 12*b*q**2

class QuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics):        
        self.t = np.full_like(ics.q, 0.72)
        
    def t_0(self, tau):
        return self.t * tau
        
    def t_1(self, tau):
        return self.t

class HalfQuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics): 
        self.t0 = ics.t.to_numpy()
        self.t = np.full_like(ics.q, 0.72/2) + self.t0
        
    def t_0(self, tau):
        return (self.t - self.t0) * tau + self.t0
        
    def t_1(self, tau):
        return self.t - self.t0
    
def eliminate_stokes(result):
    # Load projection map, map to a grid, and calculate F
    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)

    # plt.figure()
    # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

    blobs = separate_to_blobs(deriv, quantile=1e-2)
    qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]
        
    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2], 
                             time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3)
    # caustics = caustics[np.real(caustics.q) > 0]
    
    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (i, caustic) in caustics.iterrows():
        # idx = np.argmin(np.abs(proj.q0-caustic.q))
        # caustic.q = proj.q0.iat[idx]
        # caustic.xi = proj.xi.iat[idx]
        S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)
    
    return S_F

#%%

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 199), np.linspace(-2.5, 2.5, 199))
qs = (X+1j*Y).flatten()

result = propagate(create_ics(qs, S0 = [S0_0, S0_1, S0_2], gamma_f=1), 
                   V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                   time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                   blocksize=1024, n_jobs=5, verbose=True)

# x = np.arange(-12, 12, 1e-1)
# finco.show_plots(x, -1e-3, 7, 0.02, 8)

# plt.plot(x, np.abs(finco.reconstruct_psi(x, 0)))


#%%
import logging

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)
n_iters = 20
n_steps = 1
sub_tol = (1e-2,1e3)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 51), np.linspace(-2.5, 2.5, 51))
result, mesh = adaptive_sampling(qs = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2],
                                 n_iters = n_iters, sub_tol = sub_tol, plot_steps=True,
                                 V = [V_0, V_1, V_2], m = m, gamma_f = 1,
                                 time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1 / n_steps, 
                                 n_jobs=3)
         
#%%
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


#%%
trajs = load_results('trajs.hdf', gamma_f=1).get_trajectories(1)

S_F1 = eliminate_stokes(result)
plt.figure()
plt.tripcolor(np.real(trajs.q0), np.imag(trajs.q0), S_F1, shading='gouraud')
plt.colorbar()
# # plt.figure(), plt.scatter(grid.real, grid.imag, c=S_F), plt.colorbar()

x = np.linspace(-5, 5,1000)
plt.figure(), plt.plot(x, np.abs(result.reconstruct_psi(x, 1, S_F1, n_jobs=3)))

#%% Propagation continuation check
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 121), np.linspace(-2.5, 2.5, 121))
qs = (X+1j*Y).flatten()
jac = (X[0,1] - X[0,0]) *  (Y[1,0] - Y[0,0])

ics = create_ics(qs, S0 = [S0_0, S0_1, S0_2], gamma_f=1)
results1 = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f=1, jac=jac, 
                     time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                     blocksize=1024, n_jobs=1, trajs_path='full.hdf')

results2 = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f=1, jac=jac, 
                     time_traj = HalfQuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                     blocksize=1024, n_jobs=1, trajs_path='half1.hdf')

results3 = continue_propagation(results2, V = [V_0, V_1, V_2], m = m, gamma_f=1, jac=1, 
                                time_traj = HalfQuarticTimeTrajectory(), dt = 1e-3, drecord = 1, 
                                blocksize=1024, n_jobs=1, trajs_path='half2.hdf')

proj = results1.get_projection_map(1).sort_values(by='q0')

grid = np.fliplr(np.reshape(proj.q0.to_numpy(), (121, 121))).T

trajs1 = results1.get_trajectories(1,2)
trajs2 = results1.get_trajectories(2)
trajs3 = results2.get_trajectories(1)
trajs4 = results3.get_trajectories(1)

S_F1 = eliminate_stokes(results1)
S_F2 = eliminate_stokes(results3)

#%% Caustic times
import logging
logging.basicConfig()
logging.getLogger('finco').setLevel(logging.DEBUG)

def quartic_caustic_times_dist(q0, p0, t0, est):    
    return np.full_like(q0, 1e-1)

def quartic_caustic_times_dir(q0 ,p0 ,t0, est):
    return np.full_like(q0, 1)

ts = caustic_times(result, quartic_caustic_times_dir, quartic_caustic_times_dist, n_iters = 180,
                   skip = 18, x = x, plot_steps=True,
                   V = [V_0, V_1, V_2], m = m, gamma_f=1, dt=1, drecord=1, 
                   n_jobs=3, blocksize=2**15,
                   verbose=False) 