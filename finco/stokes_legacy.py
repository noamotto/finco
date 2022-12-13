# -*- coding: utf-8 -*-
"""
Legacy tools for locating caustics in FINCO results and dealing with Stokes
phenomenon. Left here as reference, but should not be used in new code.
"""

import logging
from typing import Callable, List, Tuple

from skimage import measure, feature, morphology
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import erf


def find_stokes_sectors(grid: ArrayLike, F: ArrayLike, 
                        caustic: pd.Series) -> Tuple[ArrayLike, ArrayLike]:
    """
    Locates the Stokes sectors for a specific caustic, based on image analysis
    of F.

    The function takes the sign map of the real and imaginary parts of F, and
    using image processing tools tries to divide them into the stokes and
    anti-stokes sectors. 
    
    Not so efficient, rendundant comparing to using v_t, and only works 
    for grids.

    Parameters
    ----------
    grid : ArrayLike of complex
        Grid matrix of locations of the points. Is treated as having equal
        spacing between points.
    F : ArrayLike of complex
        Calculation of the Stokes parameter F, as derived for example from
        **approximate_f**.
    caustic : pandas.Series
        The caustic to find the sectors for. Should be of the format outputted
        by **find_caustics**.

    Returns
    -------
    stokes : ArrayLike of complex
        Map of the stokes sectors, where each point's value is a sector label
        assigned to it.
    astokes : ArrayLike of complex
        Map of the anti-stokes sectors, where each point's value is a sector label
        assigned to it.
    """
    
    idx = np.argmin(np.abs(grid-caustic.q))
    x0,y0  = idx//grid.shape[1], idx % grid.shape[1]

    Fim = morphology.opening(morphology.closing(np.sign(F.imag)))
    Fim[x0-3:x0+4,y0-3:y0+4] = 0
    stokes_ = measure.label(Fim, background=0)
    s_regions = measure.regionprops(stokes_)
    rmins = [np.min([((x0 - x)**2 + (y0 - y)**2) for (x,y) in region.coords]) for
             region in s_regions]
    order = np.argsort(rmins)
    for label in order[6:]:
        stokes_[stokes_ == label + 1] = 0

    centers = np.array([np.array(s_regions[i].centroid) - (x0,y0) for i in order[:6]])
    angles = -np.angle(centers[:,1] + centers[:,0]*1j)
    newlabels = np.argsort(angles)
    oldlabels = [s_regions[i].label for i in order[:6]]

    stokes = np.zeros_like(stokes_)
    for i in range(len(newlabels)):
        stokes[stokes_ == oldlabels[newlabels[i]]] = i + 1

    Fre = morphology.opening(morphology.closing(np.sign(F.real)))
    Fre[x0-3:x0+4,y0-3:y0+4] = 0
    astokes_ = measure.label(Fre, background=0)
    a_regions = measure.regionprops(astokes_)
    rmins = [np.min([((x0 - x)**2 + (y0 - y)**2) for (x,y) in region.coords]) for
             region in a_regions]
    order = np.argsort(rmins)
    for label in order[6:]:
        astokes_[astokes_ == label + 1] = 0

    centers = np.array([np.array(a_regions[i].centroid) - (x0,y0) for i in order[:6]])
    angles = -np.angle(centers[:,1] + centers[:,0]*1j)
    newlabels = np.argsort(angles)
    oldlabels = [a_regions[i].label for i in order[:6]]

    astokes = np.zeros_like(astokes_)
    for i in range(len(newlabels)):
        astokes[astokes_ == oldlabels[newlabels[i]]] = i + 1

    return stokes, astokes

def calc_factor(F: ArrayLike, sigma: ArrayLike, stokes: ArrayLike, 
                anti_stokes: ArrayLike) -> ArrayLike:
    """
    Calculates the Barry factor given a map of stokes and anti-stokes sectors.
    
    Only works for grids, and is probably redundant comparing to using v_t.

    Parameters
    ----------
    F : ArrayLike of complex
        Calculation of the Stokes parameter F, as derived for example from
        **approximate_f**.
    sigma : ArrayLike of complex
        The sigma value for each point in the grid, as calculated by the FINCO
        propagation.
    stokes : ArrayLike of complex
        Map of the stokes sectors, where each point's value is a sector label
        assigned to it. Should be of the format returned by **find_stokes_sectors**
    anti_stokes : ArrayLike of complex
        Map of the anti-stokes sectors, where each point's value is a sector label
        assigned to it. Should be of the format returned by **find_stokes_sectors**

    Returns
    -------
    factor : ArrayLike of float in range [0,1]
        Barry factor. Should be multiplied with the prefactors of each point,
        in order to apply the treatment of Stokes phenomenon.

    """
    sigma_p = sigma > 0
    stokes_lines = feature.canny(stokes/6*100, low_threshold=1)

    factor = np.ones_like(F, dtype=np.float64)
    if len(anti_stokes[sigma_p & (anti_stokes != 0) & stokes_lines]) > 0:
        bad_label = anti_stokes[sigma_p & (anti_stokes != 0) & stokes_lines][0]
        bad_region = anti_stokes == bad_label
        s_bad_labels = np.unique(stokes[bad_region])
        fix_labels = (bad_label - 2) % 6 + 1, bad_label % 6 + 1
        fix_regions = (anti_stokes == fix_labels[0]), (anti_stokes == fix_labels[1])
        s_fix_labels = np.unique(stokes[fix_regions[0]]), np.unique(stokes[fix_regions[1]])
        sign_labels = (np.intersect1d(s_bad_labels, s_fix_labels[0]),
                       np.intersect1d(s_bad_labels, s_fix_labels[1]))

        re_sign = -np.sign(F.real[bad_region])[0]
        im_sign = [np.sign(F.imag)[stokes == sign_labels[0]][0],
                   np.sign(F.imag)[stokes == sign_labels[1]][0]]
        factor[fix_regions[0]] = (erf(re_sign * im_sign[0] * F.imag[fix_regions[0]] /
                                   (re_sign * F.real[fix_regions[0]])) + 1) / 2
        factor[fix_regions[1]] = (erf(re_sign * im_sign[1] * F.imag[fix_regions[1]] /
                                   (re_sign * F.real[fix_regions[1]])) + 1) / 2
        factor[bad_region] = 0

    return factor
