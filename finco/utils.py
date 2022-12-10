# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

# System params
hbar = 1

def gf(x, qf, pf, gamma_f):
    """
    Utility function. Reconstructs a Gaussian from given parameters.

    Parameters
    ----------
    x : 1D ArrayLike of floats
        x positions to reconstruct the Gaussian for.
    qf : float
        Gaussian's center.
    pf : float
        Gaussian's momentum.
    gamma_f : float
        Gaussian width.

    Returns
    -------
    psi : 1D Array like in the shape of x
        The reconstructed Gaussian.

    """
    X, Qf = np.meshgrid(np.array(x), np.array(qf))
    Pf = np.array(pf).reshape(-1, 1)
    return (2*gamma_f / np.pi) ** 0.25 * np.exp(-gamma_f*(X-Qf)**2 + 1j / hbar * Pf * (X-Qf))
