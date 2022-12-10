# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

from splitting_method import SplittingMethod

q = 1
keldysh = 1
omega = 7.35e-2
A0 = -omega / keldysh

def psi0(x):
    return 2*x*np.exp(-np.abs(x))

# def H_p(x):
#     cutoffRad = 750
#     cutoffStrength = 1e-3j
    
#     coulomb = -q/np.abs(x)
#     cutoff = (x > cutoffRad) * (-cutoffStrength * (x - cutoffRad)**2) + \
#         (x < -cutoffRad) * (-cutoffStrength * (x + cutoffRad)**2)
#     return coulomb + cutoff
# def psi0(x):
#     return np.exp(-(x-1)**4)

def H_p(x):
    return x**2

def H_k(p):
    return p ** 2 / 2


# halfcycle = np.pi / omega
# dt = 0.01

halfcycle = 2 * np.pi
t =  3 * 2 * halfcycle
dt = halfcycle / 500

spl = SplittingMethod(x0 = -500+1e-4, x1 = 500+1e-4, dx = 5e-2, 
                      T = t, dt = dt, trecord = halfcycle / 50, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k)

while spl.t < spl.T:
    spl.propagate()
   
spl.show_plots(0.2)
# plt.plot(spl.x, np.abs(spl.psi))
