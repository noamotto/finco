# -*- coding: utf-8 -*-
"""
PyTorch implementation of loss function based on coherent states.
"""

import torch
from torch import nn

import numpy as np

from ._utils import _calc_proj, _calc_pref, _gf

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
        
        # Create a tensor for x based on the given windows
        maxp = torch.max(torch.abs(torch.tensor([pmin, pmax])))
        x = np.arange(qmin - 4/gamma_f, qmax + 4/gamma_f + .1/maxp, .1/maxp)
        psi = np.interp(x, spl.x, self.spl.psis[-1][1])
        self.register_buffer("x",  torch.from_numpy(x))
        self.register_buffer("psi",  torch.from_numpy(psi))
        
    def forward(self, factors, trajs):
        # perform binning based on the given results
        qf, pf, _ = _calc_proj(trajs.cpu(), self.gamma_f)
        
        dq, dp = (self.qmax - self.qmin) / self.qbins, (self.pmax - self.pmin) / self.pbins
        
        qbins = ((qf - self.qmin) / dq).floor()
        pbins = ((pf - self.pmin) / dp).floor()
        bins = (qbins * self.pbins + pbins)
        
        # Keep a list of bins for forward pass
        inds_dict = {int(k): int(v) for v,k in enumerate(torch.unique(bins))}
        inds = bins.map_(bins, lambda i,x: inds_dict[int(i)]).long().to(factors.device)
        
        # Calculate ground truth for each relevant bin
        qcenters = ((qbins + 0.5) * dq + self.qmin).to(device=factors.device, dtype=self.x.dtype)
        pcenters = ((pbins + 0.5) * dp + self.pmin).to(device=factors.device, dtype=self.x.dtype)

        gfs = _gf(self.x, qcenters.flatten(), pcenters.flatten(), self.gamma_f)
        gt = torch.trapz(torch.conj(gfs) * self.psi.unsqueeze(-1), self.x, axis=0)

        # Calculate estimation based on factors and calculate loss
        pref = _calc_pref(trajs, self.gamma_f).to(factors.device)
        pref.real.nan_to_num_().clamp_(1e2, -1e2)
        pref.imag.nan_to_num_().clamp_(1e2, -1e2)
        
        vals_binned = torch.zeros_like(pref).scatter_add(1, inds, pref * factors)
        loss = (torch.abs(vals_binned - gt)**2).sum()
        
        return loss**0.5
