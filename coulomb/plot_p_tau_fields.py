# -*- coding: utf-8 -*-
"""
Plots quiver plots of the momentum field and tau-field for a given initial
condition and constant field, where tau is an estimator for the time direction
when circling the origin in position space, defined as :math:`\\tau = \\frac{imq}{p}`.

Should give intuition to how the system looks on a propagation step in oscillating
field.

The script produces three pairs of plots, one for weak field with A=1e-2, one for
medium field with A=1, and one for strong field with A=100. The plots are rescaled
a bit to make the arrows more visible, and angles are unwrapped with the initial
position as reference. In addition to the quiver plot two circles are plotted
marking the two critical radii in the system. The radius matching
    .. math:: q_+ = \\frac{-E+\\sqrt{E^2-4eA_t}}{2A_t}
is marked in blue and the radius matching
    .. math:: q_- = \\frac{-E-\\sqrt{E^2-4eA_t}}{2A_t}
is marked in orange.

@author: Noam OttolenghiTYPE
"""

#%% Setup
import numpy as np
import matplotlib.pyplot as plt

from coulombg import m, q_e

def get_p_tau(A0, q0, p0, q):
    """
    Calculates and returns the values of p(q) and tau(q) after unwrap and rescale
    for given initial conditions and field strength.

    Parameters
    ----------
    A0 : complex
        Field strength.
    q0 : complex
        Initial position.
    p0 : complex
        Initial momentum.
    q : ArrayLike of complex
        Positions to calculate for. Should be in polar form, where axis 0 (rows)
        follows the phase.

    Returns
    -------
    p : ArrayLike of complex
        The values for p(q).
    tau : ArrayLike of complex
        The values for tau(q).
    """
    E0 = p0**2/2/m - q_e/q0 - A0*q0
    K = E0 + q_e/q + A0*q
    p_angles = np.unwrap(np.angle(K), axis=0)/2
    # Additional square root taken for illustration
    p = np.abs(K**0.5/2/m) ** 0.2 * np.exp(1j*p_angles)

    tau_angles = np.unwrap(np.angle(1j*q/p), axis=0)
    tau = np.abs(q/p)**0.2 * np.exp(1j*tau_angles) # Square root taken for illustration
    return p, tau

def get_rs(A0, q0, p0):
    """
    Returns the two radii for given initial conditions and field strength.

    Parameters
    ----------
    A0 : complex
        Field strength.
    q0 : complex
        Initial position.
    p0 : complex
        Initial momentum.

    Returns
    -------
    rs : A 2-tuple of float
        The two calculated radii, with the first corresponding to
            .. math:: q_+ = \\frac{-E+\\sqrt{E^2-4eA_t}}{2A_t}
        and the second corresponding to
            .. math:: q_- = \\frac{-E-\\sqrt{E^2-4eA_t}}{2A_t}
    """
    E0 = p0**2/2/m - q_e/q0 - A0*q0
    qstars = np.concatenate([(-E0+(E0**2-4*A0*q_e)**0.5)/2/A0,
                             (-E0-(E0**2-4*A0*q_e)**0.5)/2/A0])
    return np.abs(qstars)

q0 = np.array([1-1j])
p0 = np.array([1+1j])

r = np.linspace(0.3, 4.5, 11)
theta = np.linspace(0,2,100)*np.pi + np.angle(q0)
R,T = np.meshgrid(r, theta)
q = R * np.exp(1j*T)

#%% Plot for medium field (not strong nor weak)

A0 = 1e0

rs = get_rs(A0, q0, p0)
p, tau = get_p_tau(A0, q0, p0, q)

_, (p_ax, t_ax) = plt.subplots(1,2, num=f'fields_{A0}', figsize=(8, 4.5))
plt.suptitle(f'$A_0 = {A0}$')
p_ax.quiver(q.real, q.imag, p.real, p.imag)
p_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
p_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
p_ax.set_title(r'$p(q)$')
p_ax.set_xlabel(r'$\Re q$')
p_ax.set_xlim(-2.7,2.7)
p_ax.set_ylabel(r'$\Im q$')
p_ax.set_ylim(-2.7,2.7)

t_ax.quiver(q.real, q.imag, tau.real, tau.imag)
t_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
t_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
t_ax.set_title(r'$\tau(q)$')
t_ax.set_xlabel(r'$\Re q$')
t_ax.set_xlim(-2.7,2.7)
t_ax.set_ylabel(r'$\Im q$')
t_ax.set_ylim(-2.7,2.7)

plt.tight_layout()

#%% Plot for weak field

A0 = 1e-2

rs = get_rs(A0, q0, p0)
p, tau = get_p_tau(A0, q0, p0, q)

_, (p_ax, t_ax) = plt.subplots(1,2, num=f'fields_{A0}', figsize=(8, 4.5))
plt.suptitle(f'$A_0 = {A0}$')
p_ax.quiver(q.real, q.imag, p.real, p.imag)
p_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
p_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
p_ax.set_title(r'$p(q)$')
p_ax.set_xlabel(r'$\Re q$')
p_ax.set_xlim(-2.7,2.7)
p_ax.set_ylabel(r'$\Im q$')
p_ax.set_ylim(-2.7,2.7)

t_ax.quiver(q.real, q.imag, tau.real, tau.imag)
t_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
t_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
t_ax.set_title(r'$\tau(q)$')
t_ax.set_xlabel(r'$\Re q$')
t_ax.set_xlim(-2.7,2.7)
t_ax.set_ylabel(r'$\Im q$')
t_ax.set_ylim(-2.7,2.7)

plt.tight_layout()

#%% Plot for strong field

A0 = 1e2

rs = get_rs(A0, q0, p0)
p, tau = get_p_tau(A0, q0, p0, q)

_, (p_ax, t_ax) = plt.subplots(1,2, num=f'fields_{A0}', figsize=(8, 4.5))
plt.suptitle(f'$A_0 = {A0}$')
p_ax.quiver(q.real, q.imag, p.real, p.imag)
p_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
p_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
p_ax.set_title(r'$p(q)$')
p_ax.set_xlabel(r'$\Re q$')
p_ax.set_xlim(-2.7,2.7)
p_ax.set_ylabel(r'$\Im q$')
p_ax.set_ylim(-2.7,2.7)

t_ax.quiver(q.real, q.imag, tau.real, tau.imag)
t_ax.plot(rs[0]*np.cos(theta), rs[0]*np.sin(theta))
t_ax.plot(rs[1]*np.cos(theta), rs[1]*np.sin(theta))
t_ax.set_title(r'$\tau(q)$')
t_ax.set_xlabel(r'$\Re q$')
t_ax.set_xlim(-2.7,2.7)
t_ax.set_ylabel(r'$\Im q$')
t_ax.set_ylim(-2.7,2.7)

plt.tight_layout()
