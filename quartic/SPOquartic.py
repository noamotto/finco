# -*- coding: utf-8 -*-
"""
SPO propagation for Gaussian in quartic potential. Should be used reference and
comparison to FINCO's results.

@author: Noam Ottolenghi
"""

import numpy as np
import matplotlib.pyplot as plt

from splitting_method import SplittingMethod
from quartic import S0_0, V_0

def psi0(x):
    return np.exp(1j*S0_0(x))

def H_p(x):
    return V_0(x, 0)

def H_k(p):
    return p ** 2 / 2


T = 0.72
# T = 2
dt = T / 100

spl = SplittingMethod(x0 = -50, x1 = 50, dx = 1e-2,
                      T = T, dt = dt, trecord = dt, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k)


spl.propagate()

# Show small animation of the propagation. Uncomment if needed
# spl.show_plots(0.02)
# plt.plot(spl.x, np.abs(spl.psi))
