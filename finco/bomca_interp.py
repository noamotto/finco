# -*- coding: utf-8 -*-
"""
Value interpolation for BOMCA-like propagated trajectories.

@author: Noam Ottolenghi
"""

#%% Setup
import logging
from itertools import chain
from copy import deepcopy

import numpy as np

from .mesh import Mesh
from .results import FINCOResults

class BomcaLinearInterpolator:
    """
    Interpolator for BOMCA-like propagated trajectories to use for root searches.
    Uses linear interpolation to interpolate values in set of propagated trajectories.

    Parameters
    ----------
    results : FINCOResults
        Results to interpolate from
    step : int
        Timestep to take for interpolation
    """
        
    def __init__(self, results: FINCOResults, step: int):
        # Load all necessary data
        res = results.get_results(step, step + 1)
        self.simplices = Mesh(res).tri.simplices
        self.logger = logging.Logger('finco.bomca_interp')

        # Create values for calculation
        q = np.reshape(res.q.take(self.simplices.flatten()), self.simplices.shape)
        self.A = np.stack([np.ones_like(q), np.real(q), np.imag(q)], axis=1)

    def interpolate(self, x, vals, mask, ablocks: int = 1, bblocks: int = 1):
        """
        Performs interpolation on BOMCA propagated trajectories for a set of points

        Parameters
        ----------
        x : ArrayLike of complex
            List of positions to interpolate for
        vals : ArrayLike
            Values on the propagation results for the interpolation.
            Should have one value for each trajectory.
        mask : ArrayLike of boolean
            Additional masking on the trajectories. Should be 1D array with one
            value for each simplex used for the interpolation, as held in property
            `simplices`.
        ablocks : positive integer, optional
            Number of blocks to divide the simplices list to for calculation,
            for efficient memory management. The default is 1.
        bblocks : positive integer, optional
            Number of blocks to divide the list of points to interpolate to for
            calculation, for efficient memory management. The default is 1.

        Returns
        -------
        y : list of ArrayLike
            The interpolated values for each point in x. There might be several
            values for one point in x.
        """
        def _process(inv,b,s,v,m):
            lam = inv @ b
            vals = np.reshape(v.take(s.flatten()), s.shape)
            mask = np.all(lam >= 0,axis=1) & m[:,np.newaxis]
            ys = np.einsum('tnx,tn->tx', lam, vals)
            return [y[m] for y, m in zip(ys.T, mask.T)]

        # Calculate values for interpolation and locate points in simplices
        B = np.stack([np.ones_like(x), x.real, x.imag], axis=0)
        As = np.array_split(self.A, ablocks, axis=0)
        Bs = np.array_split(B, bblocks, axis=1)
        Ss = np.array_split(self.simplices, ablocks, axis=0)
        Ms = np.array_split(mask, ablocks)

        # Perform interpolation in blocks, to reduce memory usage
        res = []
        for a,s,m in zip(As, Ss, Ms):
            inv = np.linalg.pinv(a)
            res.append(list(chain(*[_process(inv, b, s, vals, m) for b in Bs])))

        return [np.concatenate([r[i] for r in res]) for i in range(len(res[0]))]

    def __call__(self, x, vals, mask, ablocks: int = 1, bblocks: int = 1):
        """
        Shortcut for calling interpolate()
        """
        return self.interpolate(x, vals, mask, ablocks, bblocks)

###########################
#    Utility Functions    #
###########################

def get_q0s(results: FINCOResults, step: int, x: list):
    """
    Convinience function. Extracts interpolated initial positions from a propagation
    using linear interpolation

    Parameters
    ----------
    results : FINCOResults
        Results to interpolate from
    step : int
        Timestep to take for interpolation
    x : list of complex
        List of positions on the real axis to interpolate initial positions for.

    Returns
    -------
    q0s : list of lists of complex
        The interpolated initial position, where q0s[i] are the interpolated
        positions for x[i]
    """
    bomca = BomcaLinearInterpolator(results, step)
    return bomca(x, results.get_results(step, step + 1).q0)

def find_branches(q0s):
    """
    Finds continuous branches from interpolated initial positions.

    Parameters
    ----------
    q0s : list of lists of complex
        Interpolated initial position, where each sublist corresponds to the same
        position interpolated for. Should be like the result of get_q0s().

    Returns
    -------
    bs : list of lists of int
        list of branches. Each sublist corresponds to one contiuous branch, and
        contains the indices that need to be taken from each sublist of q0s to
        form the branch.
    """
    q0s = deepcopy(q0s)
    bs = []
    n = np.min([len(n) for n in q0s])
    idx = np.argmin([len(n) for n in q0s])

    for i in range(n):
        first = q0s[idx][i]
        q0s[idx][i] = np.nan
        br = [i]

        if idx > 0:
            cur = first
            for t in q0s[idx-1::-1]:
                idx_ = np.nanargmin(np.abs(cur - t))
                cur = t[idx_]
                t[idx_] = np.nan
                br.insert(0, idx_)

        if idx < len(q0s) - 1:
            cur = first
            for t in q0s[idx+1:]:
                idx_ = np.nanargmin(np.abs(cur - t))
                cur = t[idx_]
                t[idx_] = np.nan
                br.append(idx_)

        bs.append(br)
    return bs

def vals_from_branches(branches, vals):
    """
    Convenience function. Extracts the corresponding values for each branch.

    Parameters
    ----------
    branches : list of lists of int
        Branches to extract values for. Should be in the form returned by find_branches()
    vals : list of lists of complex
        Interpolated values to extract from. Should be in the form returned by
        interpolate().

    Returns
    -------
    b_vals: list of lists of complex
        The extracted values, such that b_vals[i] are the extracted values for
        the branch branches[i].
    """
    return [np.array([v[b] for (v,b) in zip(vals, br)]) for br in branches]
