# -*- coding: utf-8 -*-
"""
FINCO with adaptive sampling

This module implements FINCO propagation of a system with adaptive sampling
of the phase space.

The sampling is an iterative process, starting with building a mesh from the
initial sample given to the algorithm. The algorithm propagates the sampled
trajectories, then goes over the edges on the mesh, subsampling those fitting
the given threshold. The algorithm then propagates the newly sampled points, and
adds the new point without oversampling the branch cut. Then the algorithm considers
the newly added edges, repeating the same process.

The implementation supports different thresholds, as well as additional filtering
and plotting of intermediate steps. The algorithm stops after a given number of
steps has passed or no new points were added. The algorithm also saves snapshots
at each step, along with the new candidates to add at each step, for further analysis.

@author: Noam Ottolenghi
"""

import logging
import os
import shutil
from typing import Tuple, Union, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
from utils import tripcolor_complex
from numpy.typing import ArrayLike

from .mesh import Mesh
from .results import FINCOResults, results_from_data
from .finco import FINCOConf, create_ics, propagate, hbar


def _calc_E(u: pd.DataFrame, v: pd.DataFrame, gamma_f: float):
    """
    Calculates adaptive sampling subsampling criterion value between two points.
    The algorithm is measuring the difference between a simple finite difference
    of xi_1 and its calculated value, with respect to the calculated value (percentage).
    That way, a branch-cut causes the finite difference to diverge with respect
    to the calculated xi_1, inherently dealing with oversampling branch-cuts.

    Parameters
    ----------
    u : pandas.DataFrame
        Dataset of first points. Should contain the points' position in space
        and their xi_1 value
    v : pandas.DataFrame
        Dataset of second points. Should contain the points' position in space
        and their xi_1 value
    gamma_f : float
        Gaussian width for the Gaussians sampled for wavepacket reconstruction.

    Returns
    -------
    E: ArrayLike
        Array of calculated convergence criterion values.

    """
    u_q0, v_q0 = u.q0.to_numpy(), v.q0.to_numpy()
    u_S, v_S = u.S.to_numpy(), v.S.to_numpy()
    u_q, v_q = u.q.to_numpy(), v.q.to_numpy()
    u_p, v_p = u.p.to_numpy(), v.p.to_numpy()
    u_Z = u.Z.to_numpy()
    u_xi_1 = u.xi_1.to_numpy()
    u_dS_dq0 = u_p * u_Z

    u_Z_est = (u_q - v_q) / (u_q0 - v_q0)
    u_Pz_est = (u_p - v_p) / (u_q0 - v_q0)
    u_xi_1_est = 2 * gamma_f * u_Z_est - 1j/hbar*u_Pz_est
    u_dS_dq0_est = (u_S - v_S) / (u_q0 - v_q0)
    
    S_diff = np.abs(u_dS_dq0_est - u_dS_dq0) / np.abs(u_dS_dq0)
    xi_1_diff = np.abs(u_xi_1_est - u_xi_1) / np.abs(u_xi_1)

    pref_filter = (np.abs(u.pref.to_numpy()) > 1e-6) & (np.abs(v.pref.to_numpy()) > 1e-6)

    return (S_diff + xi_1_diff) * pref_filter

def _plot_step(mesh: Mesh, deriv: pd.DataFrame, candidate_inds: pd.DataFrame):
    """
    Plots the results of a subsampling step. The plot contains a complex color
    plot of xi_1 at the final timestep and a mesh plot of the current mesh
    shape with the following points highlighted:
        - The points rejected from subsampling altoghether as they are \
        under the threshold (green)
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
        have a 'added' field indicating whether it was accepted or rejected.
    """
    plt.figure()
    tripcolor_complex(np.real(deriv.q0), np.imag(deriv.q0), deriv.pref, absmax=1e7)
    scipy.spatial.delaunay_plot_2d(mesh.tri, plt.gca())

    rejected = candidate_inds[~candidate_inds.added]
    added = candidate_inds[candidate_inds.added]
    plt.plot(np.real(rejected.q0), np.imag(rejected.q0), 'o', c='g', ms=2, zorder=1e3)
    plt.plot(np.real(added.q0), np.imag(added.q0), 'o', c='r', ms=2, zorder=1e3)

def _get_candidates(mesh: Mesh, qs: pd.DataFrame, indices: Optional[ArrayLike] = None):
    """
    Returns a list of candidates for subsampling from points dataset and indices

    Parameters
    ----------
    mesh : Mesh
        Mesh to consider when finding neighbors.
    qs : pandas.DataFrame
        Dataset containing the points to take from. Must have the point
        coordiantes under the field 'q0'.
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
    candidates = candidates[candidates.index.get_level_values('t_index') <
                            candidates.index.get_level_values('point')]
    candidates.q0 = (candidates.q0.to_numpy() +
                    qs.take(mesh.points_to_mesh(candidates.index.get_level_values('point'))).to_numpy()) / 2

    return candidates


def _get_points_data(result: FINCOResults, n_steps: int):
    """
    Returns all the data necessary to determine the subsampling criterion. In
    practice just combines the trajectory and caustic map information into one
    dataset.

    Parameters
    ----------
    result : FINCOResults
        Results file to calculate the data from.
    n_steps : positive integer
        Number of timestep taken for each trajectory. Used to determine the final
        timestep.

    Returns
    -------
    data : pandas.DataFrame
        The combined points data.
    """
    trajs = result.get_trajectories(n_steps)
    deriv = result.get_caustics_map(n_steps)
    S = result.get_results(n_steps).S
    return (trajs.merge(deriv.drop(columns='q0'), left_index=True, right_index=True)
                 .merge(S, left_index=True, right_index=True))

def adaptive_sampling(qs, S0: list, n_iters: int,
                      sub_tol: Union[float, Tuple[float, float]],
                      filter_func: Optional[Callable[[ArrayLike], ArrayLike]] = None,
                      plot_steps: bool = False, **kwargs) -> Tuple[FINCOResults, Mesh]:
    """
    Performs FINCO propagation with adaptive sampling of the initial conditions.

    The subsampling is done by creating a mesh from the points. At every step
    the algorithm calculates the subsampling criterion as in calc_E() of the
    vertices on each new edge in the mesh, and adds a new point between them if
    it is inside the given tolerance threshold.

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
        is set to infinity.
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
    mesh : Mesh
        The resulting point mesh.
    """
    logger = logging.getLogger('finco.adaptive_sampling')

    # Check and prepare arguments
    if 'trajs_path' in kwargs and kwargs['trajs_path'] is None:
        raise NotImplementedError("Adaptive sampling does not support saving to memory")

    if isinstance(sub_tol, float):
        sub_tol = (sub_tol, np.inf)

    c = FINCOConf(**kwargs)
    n_steps = int(1/c.drecord)

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
    ics = create_ics(q0 = np.array(qs).flatten(), S0 = S0)

    logger.info('Propagating initial grid of %d points', len(ics))
    result = propagate(ics, **kwargs)
    shutil.copy(c.trajs_path, os.path.join(step_dir, "step_0.hdf"))

    res = _get_points_data(result, n_steps)
    mesh = Mesh(res, adaptive=True)

    candidate_inds = _get_candidates(mesh, res.q0)

    # Adaptive sampling iterations
    for i in range(n_iters):
        logger.info('Starting subsampling step %d/%d', i+1, n_iters)

        u = res.take(mesh.points_to_mesh(candidate_inds.index.get_level_values('t_index')))
        v = res.take(mesh.points_to_mesh(candidate_inds.index.get_level_values('point')))
        E = _calc_E(u, v, c.gamma_f)

        to_subsample = (E > sub_tol[0]) & (E < sub_tol[1])
        candidate_inds = candidate_inds[to_subsample]
        u = u[to_subsample]
        v = v[to_subsample]
        E = E[to_subsample]

        if filter_func:
            to_filter = np.array(filter_func(candidate_inds.q0))
            candidate_inds = candidate_inds[to_filter]
            E = E[to_filter]
            u = u[to_filter]
            v = v[to_filter]

        if len(candidate_inds) > 0:
            logger.info('Step %d/%d: Propagating %d candidate points',
                        i+1, n_iters, len(candidate_inds))

            # propagate candidates
            ics = create_ics(q0 = candidate_inds.q0.to_numpy(), S0 = S0)
            temp_res = propagate(ics, **temp_kwargs)

            # Calculate "energies" for subsampling
            dE = np.abs(E)

            inds_to_add = (dE > sub_tol[0]) & (dE < sub_tol[1])
            toadd = temp_res.get_results().loc[ics[inds_to_add].index.get_level_values('t_index')]
            logger.debug('<dE>=%f', np.sum(dE[inds_to_add]))

            candidate_inds['added'] = inds_to_add
            candidate_inds.to_hdf(os.path.join(step_dir, f"step_{i+1}_candidates.hdf"),
                                  'candidates')

            # Plot current step results and add new points
            if not toadd.empty:
                logger.info('Step %d added %d points', i+1, len(toadd) / (n_steps + 1))
                new_indices, toadd = mesh.add_points(toadd)
                result.merge(results_from_data(toadd, c.gamma_f))
                shutil.copy(c.trajs_path, os.path.join(step_dir, f"step_{i+1}.hdf"))
            else:
                new_indices = {}
                logger.info('Step %d added no points', i+1)

            if plot_steps:
                _plot_step(mesh, res, candidate_inds)

            # Prepare next iteration candidates
            res = _get_points_data(result, n_steps)
            new_qs = result.get_results(0,1).q0

            candidate_list = []
            new_candidates = _get_candidates(mesh, new_qs, list(new_indices.values()))
            if not new_candidates.empty:
                candidate_list.append(new_candidates)

            if candidate_list:
                candidate_inds = pd.concat(candidate_list)
            else:
                logger.info('Step %d yielded no points. Stopping', i+1)
                break
        else:
            logger.info('Step %d yielded no points. Stopping', i+1)
            break

    return result, mesh
