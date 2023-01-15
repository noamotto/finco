# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from finco import propagate, create_ics, TimeTrajectory
from utils import tripcolor_complex
from finco.stokes import separate_to_blobs, find_caustics

#%% Setup

import os

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

# System params
m = 1
omega = 0
gamma_f = 1

def S0_0(q):
    return -1j * (0.25 * np.log(2 / np.pi) - (q - 1)**4)
    
def S0_1(q):
    return 4j *(q - 1)**3

def S0_2(q):
    return 12j *(q - 1)**2

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
        return np.full_like(self.q0, 10*np.pi*tau)
    
    def t_1(self, tau):
        return np.full_like(self.q0, 10*np.pi)
    
    def get_discontinuity_times(self):
        return []

def locate_caustics_analytic(t):
    rhs = 1j * gamma_f / (6*(2*gamma_f/m*t-1j))
    return [1+rhs**0.5, 1-rhs**0.5]

#%% Run

X, Y = np.meshgrid(np.linspace(-2, 4, 61), np.linspace(-3, 3, 61))
jac = (X[0,1] - X[0,0]) *  (Y[1,0] - Y[0,0])

ics = create_ics(q0 = (X+1j*Y).flatten(), S0 = [S0_0, S0_1, S0_2], gamma_f=1)

result = propagate(ics, V = [V_0, V_1, V_2], m = m, gamma_f = gamma_f,
                   time_traj = HOTimeTrajectory(), dt = 1e-3, drecord=1/100, n_jobs=3,
                   trajs_path=None)

_, ax = plt.subplots(nrows=1, ncols=3, num='caustics-free-supergaussian', figsize=(12,4))
for i, step in enumerate([0, 20, 100]):
    caustics = locate_caustics_analytic(t=step/100*10*np.pi)
    xi_1 = result.get_caustics_map(step).xi_1
    
    plt.sca(ax[i]), tripcolor_complex(np.real(ics.q0), np.imag(ics.q0), xi_1, absmin=0.2)
    ax[i].scatter(np.real(caustics), np.imag(caustics), s=10)
    ax[i].set_xlabel(r'$\Re(q_0)$')
    ax[i].set_ylabel(r'$\Im(q_0)$')
    ax[i].set_title(r'$t={}\pi / \omega$'.format(int(step/100*10)))

plt.tight_layout()
plt.savefig('caustics-exploration/caustics-free-supergaussian.png')