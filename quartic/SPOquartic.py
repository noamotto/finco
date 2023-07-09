# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

from splitting_method import SplittingMethod

a = 0.5
b = 0.1
chi = 2j
gamma0 = 0.5

def psi0(x):
    return (2*gamma0/np.pi)**0.25 * np.exp(-gamma0 * (x-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0)

def H_p(x):
    return a*x**2 + b*x**4

def H_k(p):
    return p ** 2 / 2


# T = 0.72
T = 2
dt = T / 100
spl = SplittingMethod(x0 = -50, x1 = 50, dx = 1e-2, 
                      T = T, dt = dt, trecord = dt, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k) 


spl.propagate()
   
spl.show_plots(0.02)
# plt.plot(spl.x, np.abs(spl.psi))
