# -*- coding: utf-8 -*-
"""
General utilities
"""

from typing import Optional

import scipy.misc
from scipy.signal import correlate
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh

from splitting_method import SplittingMethod
from finco import FINCOResults

def plot_spo_vs_finco(spl: SplittingMethod, finco: FINCOResults, x: ArrayLike,
                      x0: float, x1: float, y0: float, y1: float,
                      threshold: float = -1, interval: float = 200, skip: int = 0):
    """
    Plots animation of SPO vs FINCO. Assumes both processes are of the same
    system, and took the same points in time

    Parameters
    ----------
    spl : SplittingMethod
        Propagated Splitting Operator model.
    allprops : FINCO
        Propagated FINCO model.
    x : ArrayLike
        Array of x values to reconstruct the FINCO wavefunction for
    x0 : float
        Leftmost x value
    x1 : float
        Rightmost x value
    y0 : float
        Lowest y value
    y1 : float
        Highest y value
    interval : float, optional
        Time interval between consecutive images, in milliseconds
    skip : int, optional
        Frame skip. The default is 0.

    Returns
    -------
    ani: FuncAnimation
        Animation object. Needs to be saved until animation is over..
    """
    fig, ax = plt.subplots()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    psi_spo, = plt.plot([], [])
    psi_finco, = plt.plot([], [])

    def update(frame):
        psi_spo.set_data(spl.x, np.abs(spl.psis[frame][1]))
        psi_finco.set_data(x, np.abs(finco.reconstruct_psi(x, frame, threshold)))
        return psi_spo, psi_finco

    ani = FuncAnimation(fig, update, frames = np.arange(0, len(spl.psis), skip + 1),
                        interval = interval, blit=True, repeat=False)

    return ani


def derivative(fx, dx, n=5, order=1):
    """
    Computes the numeric derivative of a function based on discrete samples,
    using the Finite Difference method. It is computed up to accuracy given by
    the number of samples and up to order given by `order`.

    Parameters
    ----------
    fx : ndarray of numbers
        Samples to calculate the derivative from. The calculation is done using
        a shifting window, as a cross-correlation.
    dx : number
        Difference between samples.
    n : int, optional
        Length of window to calculate the derivative from. Should have odd
        length, and the derivative is calculated for the middle sample.
    order : int, optional
        Order of derivative. The default is 1.

    Returns
    -------
    fd : number of type fx
    The calculated finite difference, as an approximation of the derivative.

    """
    w = scipy.misc.central_diff_weights(n, order)
    return correlate(fx, w, mode='valid') / dx**order

def complex_to_rgb(c: ArrayLike, absmin: float = 0, absmax: float = np.inf):
    """
    Converts an array of complex number to RGB, representing the number's
    norm using the color's intensity, and its phase using the color's hue.

    The color intenisty is taken from the log10 scale of the norms, with
    optional minimum and maximum cutoffs.

    Parameters
    ----------
    c : ArrayLike of complex
        complex value of each point.
    absmin : float, optional
        Minimum cutoff of the norm of complex numbers, for drawing. The default
        is 0.
    absmax : float, optional
        Maximum cutoff of the norm of complex numbers, for drawing. The default
        is np.inf.

    Returns
    -------
    rgb : ArrayLike of float of shape (c.shape, 4)
        RGBA values for each point in c.

    """
    mask = (~np.isnan(c)) & ~(np.isinf(c))
    c[~mask] = 0
    abs_c, angle_c = np.abs(c), np.angle(c)
    abs_c[abs_c < absmin] = absmin
    abs_c[abs_c > absmax] = absmax
    abs_c = np.log10(abs_c + 1e-10)
    abs_c -= np.min(abs_c)
    abs_c /= np.max(abs_c)
    rgb = hsv_to_rgb(np.stack((angle_c / 2 / np.pi + 0.5,
                               np.ones_like(abs_c),
                               abs_c), axis=-1))

    return np.concatenate((rgb, np.expand_dims(np.ones_like(abs_c), -1)),
                          axis=-1)

def tripcolor_complex(x: ArrayLike, y: ArrayLike, c: ArrayLike, 
                      triangles: Optional[ArrayLike] = None, 
                      absmin: float = 0, absmax: float = np.inf):
    """
    Creates a traingular mesh plot similar to plt.tripcolor, using complex
    numbers and complex_to_rgb().

    Parameters
    ----------
    x : ArrayLike of float
        x coordinate of points to plot.
    y : ArrayLike of float
        y coordinate of points to plot.
    c : ArrayLike of complex
        complex value of each point.
    triangles : ArrayLike, optional
        array of (ntri, 3) of point indices for vertices on the traingles to use
        in the plot. If None, then a triangluation is performed on the points.
        The default is None.
    absmin : float, optional
        Minimum cutoff of the norm of complex numbers, for drawing. The default
        is 0.
    absmax : float, optional
        Maximum cutoff of the norm of complex numbers, for drawing. The default
        is np.inf.

    Returns
    -------
    col : matplotlib.collections.Collection
        The collection representing the drawn mesh.

    """
    tri = Triangulation(x, y, triangles)
    colors = complex_to_rgb(c, absmin=absmin, absmax=absmax)

    col = TriMesh(tri, facecolors=colors)

    plt.gca().add_collection(col)
    plt.gca().autoscale_view()

    return col
