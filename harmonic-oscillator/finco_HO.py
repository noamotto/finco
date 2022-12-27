# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics, TimeTrajectory
from utils import complex_to_rgb

# System params
m = 1
omega = 1

# def S0_0(q):
#     return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**2)
    
# def S0_1(q):
#     return 2j * (q - 1)

# def S0_2(q):
#     return np.full_like(q,2j)

def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**4)
    
def S0_1(q):
    return 4j *(q - 1)**3

def S0_2(q):
    return 12j *(q - 1)**2

# def V_0(q):
#     return 0.5 * m * omega**2 * q ** 2
    
# def V_1(q):
#     return m * omega**2 * q

# def V_2(q):
#     return np.full_like(q,m * omega**2)

def V_0(q):
    return np.full_like(q, 0)
    
def V_1(q):
    return np.full_like(q, 0)

def V_2(q):
    return np.full_like(q, 0)

class HOTimeTrajectory(TimeTrajectory):
    def init(self, ics):
        self.q0 = ics.q
        
    def t_0(self, tau):
        return np.full_like(self.q0, 20*np.pi*tau)
    
    def t_1(self, tau):
        return np.full_like(self.q0, 20*np.pi)
    
    def get_discontinuity_times(self):
        return []
    
    
X, Y = np.meshgrid(np.linspace(-2, 4, 41), np.linspace(-3, 3, 61))
jac = (X[0,1] - X[0,0]) *  (Y[1,0] - Y[0,0])
# t =  2 * np.pi
# dt = t / 20

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2], gamma_f=1)

result = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f = 1,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)


# for i in range(10):
#     caustics = result.get_caustics_map(i).sort_values(by='q0')
#     grid = np.fliplr(np.reshape(caustics.q0.to_numpy(), (201, 301))).T
#     xi_1 = np.fliplr(np.reshape(caustics.xi_1.to_numpy(), (201, 301))).T
#     plt.figure(), plt.pcolormesh(grid.real, grid.imag, np.log10(np.abs((xi_1)))), plt.colorbar()
       

#%%
x = np.arange(-10, 10, 1e-2)
y = result.reconstruct_psi(x, 10)
# result.show_plots(x, -1e-3, 1, 0.02)

plt.plot(x, np.abs(y))
