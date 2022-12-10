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

def H_p(x):
    cutoffRad = 750
    cutoffStrength = 1e-3j
    
    coulomb = -q/np.abs(x)
    cutoff = (x > cutoffRad) * (-cutoffStrength * (x - cutoffRad)**2) + \
        (x < -cutoffRad) * (-cutoffStrength * (x + cutoffRad)**2)
    return coulomb + cutoff

def H_f_x(x):
    return x

def H_f_t(t):
    # return A0 * np.sin(t * omega) * np.exp(-(omega*t-np.pi/2)**2)
    return A0 * np.sin(t * omega)

def H_k(p):
    return p ** 2 / 2


halfcycle = np.pi / omega
dt = 0.0002855928413981312          # dt from werner's code. I have no clue how it was calculated though..
spl = SplittingMethod(x0 = -1280+1e-4, x1 = 1280+1e-4, dx = 2560. / 16384, 
                      T = 3 * halfcycle, dt = dt, trecord = halfcycle / 100, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k, H_f_x = H_f_x, H_f_t = H_f_t)

while spl.t < spl.T:
    spl.propagate()
   
# spl.show_plots(0.02, 5)
plt.plot(spl.x, np.abs(spl.psi))
