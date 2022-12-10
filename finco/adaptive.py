# -*- coding: utf-8 -*-
"""
The main module in FINCO. Contains an implementation of the FINCO propagation
algorithm.

The core feature of this algorithm is the ability to propagate a wavepacket
in time using semi-classical trajectories. The computation employs two types
of parallelization, by working in vectorized fashion on a set of trajectories
and an by allowing propagation using concurrent workers.

The object supports this propagation, as well as saving the results for
analysis and wavepacket reconstruction in a persistent file. It also
supports applying heuristics to the trajectories, in order to determine
problematic trajectories and discard them, through the 'heuristics' parameter.

In addition, the algorithm supports custom trajectories in time. The
algorithm propagates the system in time using a parametrized trajectory,
with parameter tau in range [0,1), and uses a provided time function t(tau)
and its first derivative (via the 't' parameter) to allow custom trajectories.

"""

import logging
import os
import shutil
from typing import Tuple, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
from utils import tripcolor_complex
from numpy.typing import ArrayLike

from .mesh import Mesh
from .results import FINCOResults, results_from_data
from .finco import FINCOConf, create_ics, propagate


def _calc_E(u: pd.DataFrame, v: pd.DataFrame):
    """
    Calculates adaptive sampling convergence criterion value between two points
    (equivalent to energy in energy minimization problems)

    Parameters
    ----------
    u : pandas.DataFrame
        Dataset of first points. Should contain the points' position in space 
        and their xi_1 value
    v : pandas.DataFrame
        Dataset of second points. Should contain the points' position in space 
        and their xi_1 value

    Returns
    -------
    E: ArrayLike
        Array of calculated convergence criterion values.

    """
    u_pref = u.pref.to_numpy()
    v_pref = v.pref.to_numpy()
    # return np.abs((u.xi_1.to_numpy() - v.xi_1.to_numpy()) /
    #               (u.q0.to_numpy() - v.q0.to_numpy()))
    # / (u_pref + v_pref)
    return (np.abs((u_pref - v_pref) / (u_pref + v_pref)) *
            ((np.abs(u_pref) > 1e-6) | (np.abs(v_pref) > 1e-6)))

def _calc_branchcut_est(u: pd.DataFrame, v: pd.DataFrame, w: pd.DataFrame):
    """
    Calculates estimator for the apperance of a branchcut. The smaller the value,
    the more likely a branchcut exists.
    
    The estimator calculates the first derivative of xi_1 in space using 2-point
    approximation and 3-point approximation using forward finite difference. 
    It is expected that if two points are from both sides of the branchcut, then
    the point in the middle between them will be in one of the sides, resulting
    in a new point very close to one of the points, and an approximation that is
    opposite in sign to the one using the two points.

    Parameters
    ----------
    u : pandas.DataFrame
        Dataset of first points. Should contain the points' position in space 
        and their xi_1 value
    v : pandas.DataFrame
        Dataset of second points. Should contain the points' position in space 
        and their xi_1 value
    w : pandas.DataFrame
        Dataset of middle points. Should contain the points' xi_1 value

    Returns
    -------
    branchcut_est : ArrayLike
        The estimator for each triplet of points.

    """
    # u_xi_1, w_xi_1, v_xi_1 = u.xi_1.to_numpy(), w.xi_1.to_numpy(), v.xi_1.to_numpy()
    # ratio = (-0.5*u_xi_1 + 0.5*v_xi_1) / (-1.5*u_xi_1 + 2*w_xi_1 - 0.5*v_xi_1)
    # return np.abs(-1 - ratio)
    return np.max([_calc_E(u, w)/_calc_E(w, v), _calc_E(w, v)/_calc_E(u, w)], axis=0)

def _plot_step(mesh: Mesh, deriv: pd.DataFrame, candidate_inds: pd.DataFrame):
    """
    Plots the results of a subsampling step. The plot contains a complex color
    plot of xi_1 at the final timestep and a mesh plot of the current mesh
    shape with the following points highlighted:
        - The points rejected from subsampling altoghether as they are \
        under the threshold (green)
        - The points that were not chosen stochastically (orage)
        - The points chosen and added (red)

    Parameters
    ----------
    mesh : Mesh
        Mesh to plot.
    deriv : pandas.DataFrame
        Dataset with the derivative map of the current step. Needed in order to
        plot the complex color plot of xi_1.
    candidate_inds : pandas.DataFrame
        Dataset with current step's candidate points. Each entry should also
        have a 'subsampled' field indicating whether it was accepted or rejected,
        and a 'added' field indicating whether it was stochsically added to the
        mesh or not.
    """
    plt.figure()
    tripcolor_complex(np.real(deriv.q0), np.imag(deriv.q0), deriv.pref)
    scipy.spatial.delaunay_plot_2d(mesh.tri, plt.gca())

    rejected = candidate_inds[~candidate_inds.subsampled]
    ignored = candidate_inds[candidate_inds.subsampled & ~candidate_inds.added]
    added = candidate_inds[candidate_inds.added]
    plt.plot(np.real(rejected.q), np.imag(rejected.q), 'o', c='g', ms=2, zorder=1e3)
    plt.plot(np.real(ignored.q), np.imag(ignored.q), 'o', c='orange', ms=2, zorder=1e3)
    plt.plot(np.real(added.q), np.imag(added.q), 'o', c='r', ms=2, zorder=1e3)

def _get_candidates(mesh: Mesh, qs: pd.DataFrame, indices: ArrayLike = None):
    """
    Returns a list of candidates for subsampling from points dataset and indices

    Parameters
    ----------
    mesh : Mesh
        Mesh to consider when finding neighbors.
    qs : pandas.DataFrame
        Dataset containing the points to take from. Must have the point
        coordiantes under the field 'q'.
    indices : ArrayLike, optional
        Array of indices of points to find candidates around. None means all
        points are considered. The default is None.

    Returns
    -------
    candidates : pandas.DataFrame
        Dataset with the candidates. Index is given as in
        Mesh.get_neighbors_value(), and the only field is 'q'.

    """
    candidates = mesh.get_neighbors_value(qs, indices)
    candidates = candidates[candidates.index.get_level_values(1) <
                            candidates.index.get_level_values(0)]
    candidates.q = (candidates.q.to_numpy() +
                    qs.take(candidates.index.get_level_values(0)).to_numpy()) / 2

    return candidates


def adaptive_sampling(qs, S0: list, n_iters: int,
                      sub_tol: Union[float, Tuple[float, float]],
                      conv_E: float, conv_N: float,
                      filter_func: Union[Callable[[ArrayLike], ArrayLike], None] = None,
                      plot_steps: bool = False, **kwargs) -> FINCOResults:
    """
    Performs FINCO propagation with adaptive sampling of the initial conditions.

    The subsampling is done by creating a mesh from the points. At every step
    the algorithm considers the absolute difference between the values of dxi_dq0
    of the vertices on each new edge in the mesh, and adds a new point between
    them if it is above the given tolerance threshold.

    The algorithms stops either after given number of steps, or when no new
    points are generated.

    Parameters
    ----------
    qs : ArrayLike of complex
        Initial positions to sample.
    S0 : ArrayLike of 3 functions
        The function S(t=0,q) and its first two spatial derivatives. Should
        be packed as [S, dS/dq, d^2S/dq^2]
    n_iters : integer
        Number of iterations to perform.
    sub_tol : float or 2-tuple of floats
        Lower and Upper tolerance thresholds for subsampling. If one number is
        given, it is treated as lower limit for subsampling, and the upper limit
        is set to the logarithm of 1/epsilon.
    conv_E : float
        DESCRIPTION.
    conv_N : float
        DESCRIPTION.
    plot_steps : bool, optional
        Whether to generate a plot with the result of the last step. The plot's
        description is documented at _plot_step()
        The default is False.
    filter_func : function of ArrayLike -> ArrayLike of bool, optional
        An optional function used to filter new points. Should return True for
        every point that should be kept, and False otherwise. If None, no points
        are filtered.

    The rest of the parameters are passed to propagate()

    Raises
    ------
    NotImplementedError
        Raised if user tries to save results to memory instead of file (not supported).

    Returns
    -------
    result : FINCOResults
        The result of the propagation.
    """
    logger = logging.getLogger('finco.adaptive_sampling')

    # Check and prepare arguments
    if 'trajs_path' in kwargs and kwargs['trajs_path'] is None:
        raise NotImplementedError("Adaptive sampling does not support saving to memory")

    if isinstance(sub_tol, float):
        sub_tol = (sub_tol, np.inf)

    c = FINCOConf(**kwargs)
    T = -np.log(np.finfo(float).eps) / sub_tol[1]
    n_steps = int(1/c.drecord)
    conv_n = 0

    # Create step folder for intermediate results
    step_dir = c.trajs_path + '.steps'

    try:
        os.mkdir(step_dir)
    except FileExistsError:
        logger.warning('Directory %s already exists. Overwriting intermediate files',
                       step_dir)

    temp_kwargs = kwargs.copy()
    temp_kwargs['trajs_path'] = os.path.join(step_dir, 'temp.hdf')

    # Initial propagation
    ics = create_ics(q0 = np.array(qs).flatten(), S0 = S0, gamma_f = c.gamma_f)
    mesh = Mesh(ics, adaptive=True)

    candidate_inds = _get_candidates(mesh, ics.q)

    logger.info('Propagating initial grid of %d points', len(ics))
    result = propagate(ics, **kwargs)
    shutil.copy(c.trajs_path, os.path.join(step_dir, "step_0.hdf"))

    # Adaptive sampling iterations
    for i in range(n_iters):
        logger.info('Starting subsampling step %d/%d', i+1, n_iters)

        deriv = result.get_trajectories(n_steps)
        u = deriv.take(candidate_inds.index.get_level_values(1))
        v = deriv.take(candidate_inds.index.get_level_values(0))
        E = _calc_E(u, v)
        
        to_subsample = (E > sub_tol[0]) & (E < sub_tol[1])
        candidate_inds = candidate_inds[to_subsample]
        u = u[to_subsample]
        v = v[to_subsample]
        E = E[to_subsample]

        if filter_func:
            to_filter = np.array(filter_func(candidate_inds.q))
            candidate_inds = candidate_inds[to_filter]
            E = E[to_filter]
            u = u[to_filter]
            v = v[to_filter]

        if len(candidate_inds) > 0:
            logger.info('Step %d/%d: Propagating %d candidate points',
                        i+1, n_iters, len(candidate_inds))

            # propagate candidates
            ics = create_ics(q0 = candidate_inds.q.to_numpy(), S0 = S0,
                             gamma_f = c.gamma_f)
            temp_res = propagate(ics, **temp_kwargs)

            # Calculate "energies" for subsampling
            w = temp_res.get_trajectories(n_steps)
            branchcut = _calc_branchcut_est(u, v, w)
            dE = np.abs(E) * branchcut

            to_subsample = (dE > sub_tol[0]) & (dE < sub_tol[1])
            inds_to_add = to_subsample
            toadd = temp_res.get_results().loc[ics[inds_to_add].index.get_level_values(0)]
            logger.debug('<dE>=%f', np.sum(dE[inds_to_add]))

            candidate_inds['subsampled'] = to_subsample
            candidate_inds['added'] = inds_to_add
            candidate_inds.to_hdf(os.path.join(step_dir, "step_{}_candidates.hdf".format(i+1)),
                                  'candidates')

            # Plot current step results and add new points
            if not toadd.empty:
                logger.info('Step %d added %d points',
                            i+1, len(toadd) / (n_steps + 1))
                new_indices, toadd = mesh.add_points(toadd)
                result.merge(results_from_data(toadd, c.gamma_f))
                shutil.copy(c.trajs_path, os.path.join(step_dir, "step_{}.hdf".format(i+1)))
            else:
                new_indices = {}
                logger.info('Step %d added no points', i+1)

            if plot_steps:
                _plot_step(mesh, deriv, candidate_inds)

            # Check convergence
            if np.abs(np.sum(dE[inds_to_add])) < conv_E:
                conv_n += 1
            else:
                conv_n = 0

            if conv_n == conv_N:
                logger.info('Adaptive sampling converged. Stopping')
                break

            # Prepare next iteration candidates
            new_qs = result.get_results(0,1).q
            old_inds = np.unique(candidate_inds.index.
                                 get_level_values(0)[to_subsample & ~inds_to_add])

            candidate_list = []
            new_candidates = _get_candidates(mesh, new_qs, list(new_indices.values()))
            if not new_candidates.empty:
                candidate_list.append(new_candidates)

            old_candidates = _get_candidates(mesh, new_qs, old_inds)
            if not old_candidates.empty:
                candidate_list.append(old_candidates)

            if candidate_list:
                candidate_inds = pd.concat(candidate_list)
            else:
                logger.info('Step %d yielded no points. Stopping', i+1)
                break
        else:
            logger.info('Step %d yielded no points. Stopping', i+1)
            break

    return result, mesh
