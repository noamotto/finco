# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Setup
from .mesh import Mesh
from .results import FINCOResults
import numpy as np
import logging

class BomcaLinearInterpolator:
    def __init__(self, results: FINCOResults, step: int, gamma_f: float):
        # Load all necessary data
        res = results.get_results(step, step + 1)
        self.simplices = Mesh(res).tri.simplices
        self.logger = logging.Logger('finco.bomca_interp')
        
        # Create values for calculation
        q = np.reshape(res.q.take(self.simplices.flatten()), self.simplices.shape)
        self.A = np.stack([np.ones_like(q), np.real(q), np.imag(q)], axis=1)
        
    
    def interpolate(self, x, val):
        """
        Performs interpolation according to BOMCA for a set of points

        Parameters
        ----------
        x : ArrayLike of real values
            Set of points for the interpolation
        val : ArrayLike
            Values on the propagation results for the interpolation. 
            Should have one value for each trajectory.

        Returns
        -------
        y : list of ArrayLike
            The interpolated values for each point in x. There might be several
            values for one point in x.

        """
        if np.any(np.iscomplex(x)):
            raise TypeError('At least one input point is complex.')
        
        # Calculate values for interpolation and locate points in simplices
        B = np.stack([np.ones_like(x),x,np.zeros_like(x)], axis=0)
        lambdas = np.linalg.pinv(self.A) @ B
        vals = np.reshape(np.take(val, self.simplices.flatten()), self.simplices.shape)
        mask = np.all(lambdas >= 0,axis=1)
        ys = np.einsum('tnx,tn->tx', lambdas, vals)
        return [y[m] for y, m in zip(ys.T, mask.T)]
    
    def __call__(self, x, val):
        """
        Shortcut for calling interpolate()
        """
        return self.interpolate(x, val)
