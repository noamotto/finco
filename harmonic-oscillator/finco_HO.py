# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

from finco import propagate, create_ics, TimeTrajectory
from utils import tripcolor_complex

#%% Setup

# System params
m = 1
omega = 1

def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**2)
    
def S0_1(q):
    return 2j * (q - 1)

def S0_2(q):
    return np.full_like(q,2j)

def V_0(q):
    return 0.5 * m * omega**2 * q ** 2
    
def V_1(q):
    return m * omega**2 * q

def V_2(q):
    return np.full_like(q,m * omega**2)

class HOTimeTrajectory(TimeTrajectory):
    def init(self, ics):
        self.q0 = ics.q
        
    def t_0(self, tau):
        return np.full_like(self.q0, 2*np.pi*tau)
    
    def t_1(self, tau):
        return np.full_like(self.q0, 2*np.pi)
    
    def get_discontinuity_times(self):
        return []
    
#%% Run
X, Y = np.meshgrid(np.linspace(-2, 4, 41), np.linspace(-3, 3, 61))

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2], gamma_f=1)

result = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f = 1,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)       

x = np.arange(-10, 10, 1e-2)
y = result.reconstruct_psi(x, 100)

plt.plot(x, np.abs(y))
