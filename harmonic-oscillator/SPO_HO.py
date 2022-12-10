# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

from splitting_method import SplittingMethod

def psi0(x):
    return (2/np.pi)**0.25 * np.exp(-(x-1)**2)

def H_p(x):
    return x**2 / 2

def H_k(p):
    return p ** 2 / 2


dt = np.pi * 2 / 100
spl = SplittingMethod(x0 = -50, x1 = 50, dx = 1e-2, 
                      T = np.pi * 2, dt = dt, trecord = dt, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k) 


spl.propagate()
   
spl.show_plots(0.02)
# plt.plot(spl.x, np.abs(spl.psi))
