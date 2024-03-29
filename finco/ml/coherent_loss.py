# -*- coding: utf-8 -*-
"""
Loss function based on coherent states
"""

from itertools import product
import numpy as np
import pandas as pd

from finco.results import gf

class CoherentLoss:
    def __init__(self, qbins, pbins, gamma_f = 1, dy = 1e-3):
        self.qbins = qbins
        self.pbins = pbins
        self.gamma_f = gamma_f
        
    def init(self, res, spl):
        # perform binning based on the given results
        xi = 2 * self.gamma_f * res.q - 1j * res.p
        qf, pf = np.real(xi) / 2 / self.gamma_f, -np.imag(xi)
        
        qbinned = pd.Series(qf).groupby(pd.cut(qf, self.qbins))
        pbinned = pd.Series(pf).groupby(pd.cut(pf, self.pbins))

        self.binned = [list(set(qbinned.groups[q]) & set(pbinned.groups[p])) 
                       for q,p in product(qbinned.groups.keys(), pbinned.groups.keys())]
        
        # Calculate ground truth foreach relevant bin
        qcenters = (self.qbins[:-1] + self.qbins[1:])/2
        pcenters = (self.pbins[:-1] + self.pbins[1:])/2
        relevant = np.where(np.reshape([bool(i) for i in self.binned], 
                                       (self.qbins.size - 1, self.pbins.size - 1)))
        self.qs = qcenters[relevant[0]]
        self.ps = pcenters[relevant[1]]
        gfs = gf(spl.x, self.qs, self.ps, self.gamma_f)
        self.gt = np.trapz(np.conj(gfs) * spl.psis[-1][1], spl.x, axis=1)

    def forward(self, factors, trajs, deriv):
        relevant = np.where([bool(i) for i in self.binned])[0]
        vals = np.nan_to_num(trajs.pref) / np.abs(deriv.xi_1)**2 * factors
        vals_binned = [vals.iloc[self.binned[i]].sum() for i in relevant]
        self.loss = np.sum(np.abs(vals_binned - self.gt)**2)
        
        return self.loss
    
    def backward(self, factors, trajs, deriv):
        # Find the bin for each trajectory
        bins = np.full(len(trajs), -1)
        relevant = np.where([bool(i) for i in self.binned])[0]
        for i,b in enumerate(relevant):
            bins[self.binned[relevant[i]]] = i
        
        # Calculate necessary quantities
        vals = np.nan_to_num(trajs.pref) / np.abs(deriv.xi_1)**2 * factors
        vals_binned = [(vals * factors).iloc[self.binned[i]].sum() for i in relevant] - self.gt
        
        inter = vals*np.conj(vals_binned.take(bins))
        return (inter + np.conj(inter)) * (bins != -1)
        