# -*- coding: utf-8 -*-
"""
PyTorch implementation of loss function based on coherent states.
"""

import torch
from torch import nn

from ._utils import _calc_proj, _calc_pref
from ...results import gf

class CoherentLoss(nn.Module):
    def __init__(self, qmin, qmax, qbins, pmin, pmax, pbins, spl, gamma_f = 1):
        super(CoherentLoss, self).__init__()
        
        self.qmin = qmin
        self.qmax = qmax
        self.qbins = qbins
        self.pmin = pmin
        self.pmax = pmax
        self.pbins = pbins
        self.spl = spl
        self.gamma_f = gamma_f
        
    def forward(self, factors, trajs):
        # perform binning based on the given results
        qf, pf, _ = _calc_proj(trajs, self.gamma_f)
        
        dq, dp = (self.qmax - self.qmin) / self.qbins, (self.pmax - self.pmin) / self.pbins
        
        qbins = ((qf - self.qmin) / dq).floor()
        pbins = ((pf - self.pmin) / dp).floor()
        bins = qbins * self.pbins + pbins
        
        # Keep a list of bins for forward pass
        inds_dict = {int(k): int(v) for v,k in enumerate(torch.unique(bins))}
        inds = bins.clone().map_(bins, lambda i,x: inds_dict[int(i)]).long()
        
        # Calculate ground truth for each relevant bin
        qcenters = (qbins + 0.5) * dq + self.qmin
        pcenters = (pbins + 0.5) * dp + self.pmin

        gfs = torch.from_numpy(gf(self.spl.x, qcenters, pcenters, self.gamma_f))
        gt = torch.trapz(torch.conj(gfs) * torch.from_numpy(self.spl.psis[-1][1]),
                         torch.from_numpy(self.spl.x), axis=1).to(factors.device)

        # Calculate estimation based on factors and calculate loss
        pref = _calc_pref(trajs, self.gamma_f).to(factors.device)
        pref.real.nan_to_num_().clamp_(1e2, -1e2)
        pref.imag.nan_to_num_().clamp_(1e2, -1e2)
        
        vals_binned = torch.zeros_like(pref).scatter_add(1, inds, pref * factors)
        loss = (torch.abs(vals_binned - gt)**2).sum()
        
        return loss**0.5
