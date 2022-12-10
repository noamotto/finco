# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

# System params
hbar = 1

class Heuristic:
    """Baseclass for FINCO trajectory heuristics
    
    The purpose of these heuristics is to determine which trajectories 
    should be thrown. It is used in FINCO's propagation to throw invalid 
    trajectories.
    
    Call parameters
    ---------------
    trajs : dict
        A dictionary with full trajectory data. The data is packaged as 
        matrices of shape (# trajectories, # timesteps), where the element 
        at cell (trajectory, timestep) contains the data for that trajectory 
        and timestep.
        
        Contains the following fields:
            
        - q0: Initial positions, 
        - p0: Initial momenta
        - q: Positions at each timestep 
        - p: Momenta at each timestep
        - S: S_cl at each timestep
        - Mq,Mp: Stability matrix elements at each timestep
        - qf: Center of the reconstructed Gaussian
        - pf: Momentum of the reconstructed Gaussian
        - pref: Prefactor of the reconstructed Gaussian
    
    Returns
    -------
    mask : ArrayLike of shape (# trajectories, # steps)
        Array of masks for the tajectories, where trajectories that should 
        be thrown are masked with False (or equivalent)

    """
    def __call__(self, trajs):
        raise NotImplementedError


class NonDivergeingSigmaHeuristic(Heuristic):
    """
    Heuristic for non-diverging trajectories based on their sigma. Throws 
    every trajectory with positive sigma.real
    
    Init parameters
    ---------------
    gamma_f : float
        Width of the reconstructed Gaussian. Used for calculating sigma.
    
    Call parameters
    ---------------
    trajs : dict
        A dictionary with full trajectory data. The data is packaged as 
        matrices of shape (# trajectories, # timesteps), where the element 
        at cell (trajectory, timestep) contains the data for that trajectory 
        and timestep.
        
        Contains the following fields:
            
        - q0: Initial positions, 
        - p0: Initial momenta
        - q: Positions at each timestep 
        - p: Momenta at each timestep
        - S: S_cl at each timestep
        - Mq,Mp: Stability matrix elements at each timestep
        - qf: Center of the reconstructed Gaussian
        - pf: Momentum of the reconstructed Gaussian
        - pref: Prefactor of the reconstructed Gaussian
    
    Returns
    -------
    mask : ArrayLike of shape (# trajectories, # steps)
        Array of masks for the tajectories, where trajectories that should 
        be thrown are masked with False (or equivalent)
    """
    
    def __init__(self, gamma_f):
        self.gamma_f = gamma_f
    
    def __call__(self, trajs):
        p, pf, S = trajs['p'], trajs['pf'], trajs['S']
        sigma = 1j / hbar * S + (p**2 - pf ** 2) / 4 / self.gamma_f / hbar ** 2
        return sigma.real <= 0


class NonDivergeingMqHeuristic(Heuristic):
    """
    Heuristic for non-diverging trajectories based on their Mq. Throws 
    every trajectory with absolute value of Mqq above given threshold.
    
    Init parameters
    ---------------
    threshold : float
        Threshold for masking.
    
    Call parameters
    ---------------
    trajs : dict
        A dictionary with full trajectory data. The data is packaged as 
        matrices of shape (# trajectories, # timesteps), where the element 
        at cell (trajectory, timestep) contains the data for that trajectory 
        and timestep.
        
        Contains the following fields:
            
        - q0: Initial positions, 
        - p0: Initial momenta
        - q: Positions at each timestep 
        - p: Momenta at each timestep
        - S: S_cl at each timestep
        - Mq,Mp: Stability matrix elements at each timestep
        - qf: Center of the reconstructed Gaussian
        - pf: Momentum of the reconstructed Gaussian
        - pref: Prefactor of the reconstructed Gaussian
    
    Returns
    -------
    mask : ArrayLike of shape (# trajectories, # steps)
        Array of masks for the tajectories, where trajectories that should 
        be thrown are masked with False (or equivalent)
    """
    
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, trajs):
        Mq = trajs['Mq']
        return np.abs(Mq) < self.threshold

