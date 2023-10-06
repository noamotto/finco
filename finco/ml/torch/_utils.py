# -*- coding: utf-8 -*-
"""
Implementation of FINCO functions in PyTorch
"""

import torch

def _gf(x, qf, pf, gamma_f):
    """
    Utility function. Reconstructs a Gaussian from given parameters.

    Parameters
    ----------
    x : 1D ArrayLike of floats
        x positions to reconstruct the Gaussian for.
    qf : float
        Gaussian's center.
    pf : float
        Gaussian's momentum.
    gamma_f : float
        Gaussian width.

    Returns
    -------
    psi : 1D Array like in the shape of x
        The reconstructed Gaussian.

    """
    X, Qf = torch.meshgrid(x, qf)
    Pf = pf.unsqueeze(0)
    return (2*gamma_f / torch.pi) ** 0.25 * torch.exp(-gamma_f*(X-Qf)**2 + 1j * Pf * (X-Qf))

def _calc_proj(x, gamma_f):
    """
    Calculates the projection parameters of given trajectories.

    Parameters
    ----------
    x : torch.tensor
        Tensor contining the trajectories, as returned by FINCODataset. The 
        tensor should have the following shape NxTx11x2 where N is the number
        of trajectories and T is the number of timesteps taken.
    gamma_f : float
        Parameter dictating the used coherent basis.

    Returns
    -------
    qf : torch.tensor of shape NxT
        The positions of the coherent states projected by each trajectory at
        each time.
    pf : torch.tensor of shape NxT
        The momenta of the coherent states projected by each trajectory at
        each time.
    sigma : torch.tensor of shape NxT
        The exponent part of the prefactor of the coherent states projected by
        each trajectory at each time.
    """
    q = torch.view_as_complex(x[:,:,4,:])
    p = torch.view_as_complex(x[:,:,5,:])
    S = torch.view_as_complex(x[:,:,6,:])
    xi = 2 * gamma_f * q - 1j * p
    qf, pf = xi.real / 2 / gamma_f, -xi.imag
    sigma = 1j * S + (p**2 - pf**2) / 4 / gamma_f
    return qf, pf, sigma

def _calc_xi_1(x, gamma_f):
    """
    Calculates the derivative parameters of given trajectories.
    
    Parameters
    ----------
    x : torch.tensor
        Tensor contining the trajectories, as returned by FINCODataset. The 
        tensor should have the following shape NxTx11x2 where N is the number
        of trajectories and T is the number of timesteps taken.
    gamma_f : float
        Parameter dictating the used coherent basis.

    Returns
    -------
    xi_1 : torch.tensor of shape NxT
        The value of dxi/dq0 of each trajectory at each time.
    Z : torch.tensor of shape NxT
        The value of dq/dq0 of each trajectory at each time.
    Pz : torch.tensor of shape NxT
        The value of dp/dq0 of each trajectory at each time.
    """
    S_20 = torch.view_as_complex(x[:,:,2,:])
    Mqq = torch.view_as_complex(x[:,:,7,:])
    Mqp = torch.view_as_complex(x[:,:,8,:])
    Mpq = torch.view_as_complex(x[:,:,9,:])
    Mpp = torch.view_as_complex(x[:,:,10,:])
    Z, Pz = Mqq + Mqp * S_20, Mpq + Mpp * S_20
    xi_1 = 2 * gamma_f * Z - 1j * Pz
    return xi_1, Z, Pz

def _calc_pref(x, gamma_f):
    """
    Calculates the prefactor (the estimated overlap integral) of given trajectories.
    
    Parameters
    ----------
    x : torch.tensor
        Tensor contining the trajectories, as returned by FINCODataset. The 
        tensor should have the following shape NxTx11x2 where N is the number
        of trajectories and T is the number of timesteps taken.
    gamma_f : float
        Parameter dictating the used coherent basis.

    Returns
    -------
    pref : torch.tensor of shape NxT
        The prefactor (the estimated overlap integral) of each trajectory at each time.
    """
    _, _, sigma = _calc_proj(x, gamma_f)
    xi_1, _, _ = _calc_xi_1(x, gamma_f)
    
    return (2 * gamma_f * torch.pi) ** 0.25 * xi_1 ** (-0.5) * torch.exp(sigma)
