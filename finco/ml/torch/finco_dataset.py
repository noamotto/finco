# -*- coding: utf-8 -*-
"""
Pytorch dataset wrapper for a FINCO results dataset. Wraps a dataset to return
a torch.tensor of trajectories at a given range ot timesteps.
"""

from ...results import FINCOResults

from itertools import product
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FINCODataset(Dataset):
    def __init__(self, result: FINCOResults, qfmin, qfmax, qbins, pfmin, pfmax, pbins, step):
        self.result = result.get_results(step, step + 1)
            
        xi = 2 * result.gamma_f * self.result.q - 1j * self.result.p
        qf, pf = np.real(xi) / 2 / result.gamma_f, -np.imag(xi)        
        mask = (qf > qfmin) & (qf < qfmax) & (pf > pfmin) & (pf < pfmax)
        self.result = self.result[mask]
        qf = qf[mask]
        pf = pf[mask]
        
        qbinned = pd.Series(qf).groupby(pd.cut(qf, qbins))
        pbinned = pd.Series(pf).groupby(pd.cut(pf, pbins))

        # Get a list of indices to choose from by filtering the empty bins
        binned = [list(set(qbinned.groups[q]) & set(pbinned.groups[p])) 
                  for q,p in product(qbinned.groups.keys(), pbinned.groups.keys())]
        self.bins = list(filter(lambda x: bool(x), binned))
        
        # Give a reasonable number of possible permutations (in practice we will have much more)
        self.count = int(np.count_nonzero([len(x) for x in binned]) *
                         np.mean([len(x) for x in binned]))
        self.samples = np.random.randint(np.zeros((len(self.bins), self.count)),
                                         np.array([len(x) for x in self.bins])[:,None])

    def __len__(self):
        return self.count
    
    @property
    def n_trajs(self):
        return len(self.bins)
    
    def __getitem__(self, index):
        samples = [self.bins[i][s] for i, s in enumerate(self.samples[:,index])]
        idxs = self.result.index.get_level_values('t_index')[samples]
        vals = self.result.loc[idxs,:,:].to_numpy()
        
        return torch.FloatTensor(np.stack([vals.real, vals.imag], axis=-1))
