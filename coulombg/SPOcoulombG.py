# -*- coding: utf-8 -*-
"""
SPO propagation for Gaussian in quartic potential. Should be used reference and
comparison to FINCO's results.

@author: Noam Ottolenghi
"""

import numpy as np

from splitting_method import SplittingMethod

q_e = 1

def psi0(x):
    return 2*x*np.exp(-np.abs(x))

def H_p(x):
    cutoff_rad = 750
    cutoff_strength = 1e-3j
    
    coulomb = -q_e/np.abs(x)
    cutoff = (x > cutoff_rad) * (-cutoff_strength * (x - cutoff_rad)**2) + \
        (x < -cutoff_rad) * (-cutoff_strength * (x + cutoff_rad)**2)
    return coulomb + cutoff

def H_k(p):
    return p ** 2 / 2

halfcycle = 2 * np.pi
t =  3 * 2 * halfcycle
dt = halfcycle / 500

spl = SplittingMethod(x0 = -500+1e-4, x1 = 500+1e-4, dx = 5e-2, 
                      T = t, dt = dt, trecord = halfcycle / 50, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k)

while spl.t < spl.T:
    spl.propagate()
   
spl.show_plots(0.2)
