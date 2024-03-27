# -*- coding: utf-8 -*-
"""
Tools for locating caustics in FINCO results and dealing with Stokes phenomenon.
"""

import logging
import os
import shutil
from time import perf_counter
from typing import Callable, List, Tuple, Optional

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import erf
from utils import derivative
from joblib import Parallel, delayed

from .finco import propagate, create_ics, continue_propagation
from .results import FINCOResults, get_view
from .time_traj import TimeTrajectory, LineTraj
from .mesh import Mesh

def separate_to_blobs(deriv: pd.DataFrame, quantile: float = 1e-3,
                      connectivity: int = 2) -> List[set]:
    """
    Separates a mesh of results into blobs of possible locations of caustics.
    The process is done by taking the points whose absolute value of dxi_dq0 is
    lower than the given quantile, and then separating them to blobs using the
    connectivity over some triangulated mesh.

    Parameters
    ----------
    deriv : pandas.DataFrame
        Dataset of caustic data from results, as created by
        FINCOResults.get_caustics_map()
    quantile : float in [0,1], optional
        Quantile for candidate location. The default is 1e-3.
    connectivity : positive int, optional
        Maximal number of hops on the mesh allowed for a point to be
        considered a neighbor. For example, connectivity of 1 considers only
        the points connected directly to the given point as neighbors, while
        connectivity of 2 considers the points connected to those points as
        neighbors as well. The default is 1.

    Returns
    -------
    blobs : list of sets
        List of found blobs with possible caustics.

    """
    mesh = Mesh(deriv)
    candidates = set(deriv[np.abs(deriv.xi_1) <
                           deriv.xi_1.abs().quantile(quantile)].index.get_level_values(0))
    blobs = []

    while candidates:
        cur = candidates.pop()
        blobs.append({cur})
        neighbors = candidates & mesh.get_neighbors(cur, connectivity)

        while neighbors:
            blobs[-1] |= neighbors
            candidates -= neighbors
            neighbors = (candidates &
                         set().union(*[mesh.get_neighbors(i, connectivity) for i in neighbors]))

    return [list(blob) for blob in blobs]

def find_caustics(qs: ArrayLike, S0: ArrayLike, progress: bool = True,
                  threshold: float = 1e-2, **kwargs) -> pd.DataFrame:
    """
    Finds the caustics of a system given the parameters for FINCO propagation
    and initial guesses.

    The algorithm uses a root-search to find the caustics, and therefore is
    quite expensive and should not be used too sparingly.

    After finding candidates the algorithm goes through the points and
    eliminates duplicates, returning the point with minimal value for each
    cluster of close points.

    Parameters
    ----------
    qs : ArrayLike of complex
        Initial guesses for the root search. Each guess yields one condidate
        for being a caustic.
    S0 : ArrayLike of 3 functions
        The function S(t=0,q) and its first two spatial derivatives. Should
        be packed as [S, dS/dq, d^2S/dq^2]
    progress : bool, optional
        Show progress over the initial guesses. The default is True.
    threshold : float, optional
        Distance of xi_1 from zero from which to treat point as a caustics.
        The default is 1e-2.

    Returns
    -------
    caustics: pandas.DataFrame
        Dataframe with the found caustics. Consists of the following fields:

            - q : complex
                The found point
            - dxi_f_dq0 : complex
                The value of dxi_f/dq0 at the caustic
            - norm : float
                The norm of dxi_f_dq0
    """
    def run_finco(q):
        results = propagate(create_ics(np.array([q[0]+q[1]*1j]), S0=S0), **kwargs)

        res = results.get_caustics_map(step=1).xi_1.to_numpy()
        return float(res.real), float(res.imag)

    kwargs['drecord'] = 1
    kwargs['verbose'] = False
    kwargs['trajs_path'] = None
    n_jobs = kwargs.pop('n_jobs', 1)

    res = Parallel(n_jobs=n_jobs, verbose=10 * progress)([delayed(root)(run_finco,
                                                                        x0=(x,y)) for x, y
                                                          in zip(np.real(qs), np.imag(qs))])

    converged = [r.success for r in res]
    roots = np.array([r.x for r in res])[converged]
    vals = np.array([r.fun for r in res])[converged]
    candidates = pd.DataFrame({'q': roots[:,0] + roots[:,1]*1j,
                               'xi_1': vals[:,0] + vals[:,1]*1j})

    # Calculate norm and throw candidates above threshold
    candidates['norm'] = np.abs(candidates['xi_1'])
    candidates = candidates[candidates.norm < threshold].reset_index(drop=True)

    caustics = []
    while not candidates.empty:
        val = candidates.q[0]
        close = np.isclose(val, candidates.q)
        cluster = candidates[close]
        idx = cluster.idxmin().norm
        caustics.append(pd.DataFrame(candidates.iloc[idx]).T)
        candidates = candidates[~close].reset_index(drop=True)

    caustics = pd.concat(caustics, ignore_index=True)

    # Find derivatives around caustics
    for i, caustic in caustics.iterrows():
        X, Y = np.meshgrid(np.linspace(np.real(caustic.q - 1e-3), np.real(caustic.q + 1e-3), 5),
                           np.imag(caustic.q))
        window = propagate(create_ics((X+1j*Y).flatten(), S0=S0), **kwargs)
        deriv = window.get_caustics_map(1)
        caustics.loc[i, 'xi'] = window.get_projection_map(1).iloc[2].xi

        dx = deriv.q0.diff().iloc[1]
        caustics.loc[i, 'xi_2'] = derivative(deriv.xi_1, dx)
        caustics.loc[i, 'xi_3'] = derivative(deriv.xi_1, dx, order=2)
        caustics.loc[i, 'sigma_2'] = derivative(deriv.sigma_1, dx)
        caustics.loc[i, 'sigma_3'] = derivative(deriv.sigma_1, dx, order=2)

    return caustics

def approximate_F(q0: pd.Series, xi: pd.Series,
                  caustic: pd.Series) -> Tuple[pd.DataFrame, complex]:
    """
    Calculates the approximation of the Stokes parameter :math:`F` as described in
    https://aip.scitation.org/doi/pdf/10.1063/1.5024467

    Parameters
    ----------
    q0 : pandas.Series of complex
        Series of initial positions, with index as in FINCO's results and
        create_ics()
    xi : pandas.Series of complex
        Series of values of the projection map corresponding to the initial
        positions, with index as in FINCO's results and create_ics()
    caustic : pandas.Series
        A caustic to compute F for. Should be of the format outputted by
        find_caustics()

    Returns
    -------
    F : pandas.DataFrame
        Approximation of :math:`F` corresponding to the initial positions. Contains the
        same index as q0 and the following fields:
            - F: complex
                The approximation of `F` for each point
            - v_t: complex
                The calculated value of :math:`\\tilde\\nu` for each point
    F_3 : complex
        The calculated value of :math:`F^{(3)}`
    """
    v_t = np.stack([((xi - caustic.xi) * 2 / caustic.xi_2)**0.5,
                    -((xi - caustic.xi) * 2 / caustic.xi_2)**0.5])
    sign = np.argmin(np.abs(v_t - (q0 - caustic.q).to_numpy()[np.newaxis,:]), axis=0)
    v_t = np.take_along_axis(v_t, sign[np.newaxis, :], 0).squeeze()
    F_3 = caustic.sigma_3 / 3 - caustic.sigma_2 * caustic.xi_3 / 3 / caustic.xi_2
    return pd.DataFrame({'F': F_3 * v_t ** 3, 'v_t': v_t}, index = q0.index), F_3

def calc_factor2(caustic: pd.Series, q0: pd.Series, xi: pd.Series,
                 sigma: pd.Series) -> pd.Series:
    """
    Calculates the Berry factor using :math:`\\tilde\\nu` for a caustic, as described in
    https://aip.scitation.org/doi/pdf/10.1063/1.5024467

    The method calculates :math:`\\tilde\\nu` using the caustic, and the
    :math:`\\xi` and :math:`\\sigma` values on
    the plane, infers from it the approximation of the Stokes and anti-Stokes
    sectors, and calculates the full Berry factor. The method is very efficient,
    but might need better treatment of the time trajectories.

    Parameters
    ----------
    caustic : pandas.Series
        The caustic to calculate the factor for. Should be of the format outputted
        by find_caustics().
    q0 : pandas.Series of complex
        Series of initial positions, with index as in FINCO's results and
        create_ics()
    xi : pandas.Series of complex
        Series of :math:`\\xi` values corresponding to the initial
        positions, with index as in FINCO's results and create_ics()
    sigma : pandas.Series of complex
        Series of :math:`\\sigma` values corresponding to the initial
        positions, with index as in FINCO's results and create_ics()

    Returns
    -------
    factor : ArrayLike of float in range [0,1]
        Berry factor. Should be multiplied with the prefactors of each point,
        in order to apply the treatment of Stokes phenomenon.

    """
    logger = logging.getLogger('finco.calc_factor2')

    factor = np.ones_like(xi, dtype=np.float64)
    eps = np.finfo(factor.dtype).eps

    F, F_3 = approximate_F(q0, xi, caustic)
    phi0 = np.angle(F_3)
    phis = (np.arange(-4,4) * np.pi - phi0)/3
    phis = phis[(phis > -np.pi) & (phis < np.pi)]

    # Find the point closest in angle to a Stokes line, preferring points closer
    # to the caustic. In addition, we restrict ourselves only to points close
    # to the caustic
    divergent_mask = np.real(sigma) > 0

    # Determine radius of r. If using F_4 / F_3 yields nothing, take a radius of
    # very low percentile of points.
    r = np.abs(-caustic.xi_2*2/caustic.xi_3)
    if not (divergent_mask & (F.v_t.abs() <= r)).any():
        # We can't deal with this caustic then...
        return pd.Series(factor, index=q0.index)
        # r = F.v_t[divergent_mask].abs().quantile(1e-3)
    divergent_mask &= (F.v_t.abs() < r).to_numpy()

    dists = np.angle(F.v_t[divergent_mask])[:,np.newaxis] - phis[np.newaxis,:]
    dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)
    candidates = [np.min(F.v_t[divergent_mask][(dists[:,i] < 1e-1)].abs()) if not
                  F.v_t[divergent_mask][(dists[:,i] < 1e-1)].empty else np.nan for
                  i in range(6)]
    if np.all(np.isnan(candidates)):
        logger.info("No nonphyscal region was found at anti-Stokes line. Skipping")
        return pd.Series(factor, index=q0.index)
    angle = np.nanargmin(candidates)

    n = np.round((3*phis[angle]+phi0) / np.pi)
    re_sign = (-1)**(n+1)
    im_sign = [(-1)**n, -(-1)**n]

    bad_region = ((np.abs(np.angle(F.v_t) - phis[angle]) < np.pi/6) |
                  (np.abs(np.angle(F.v_t) - phis[angle] + 2*np.pi) < np.pi/6) |
                  (np.abs(np.angle(F.v_t) - phis[angle] - 2*np.pi) < np.pi/6))
    fix_regions = [((np.angle(F.v_t) - phis[angle] > -np.pi/2) &
                    (np.angle(F.v_t) - phis[angle] < -np.pi/6) |
                    ((np.angle(F.v_t) - phis[angle] > 3*np.pi/2) &
                     (np.angle(F.v_t) - phis[angle] < 11*np.pi/6)) |
                    ((np.angle(F.v_t) - phis[angle] > -5*np.pi/2) &
                     (np.angle(F.v_t) - phis[angle] < -13*np.pi/6))),
                   ((np.angle(F.v_t) - phis[angle] < np.pi/2) &
                    (np.angle(F.v_t) - phis[angle] > np.pi/6)) |
                   ((np.angle(F.v_t) - phis[angle] < 5*np.pi/2) &
                    (np.angle(F.v_t) - phis[angle] > 13*np.pi/6)) |
                   ((np.angle(F.v_t) - phis[angle] < -3*np.pi/2) &
                    (np.angle(F.v_t) - phis[angle] > -11*np.pi/6))]
    factor[fix_regions[0]] *= (erf(im_sign[0] * np.imag(F.F)[fix_regions[0]] /
                               ((2 * re_sign * np.real(F.F)[fix_regions[0]])**0.5 + eps)) + 1) / 2
    factor[fix_regions[1]] *= (erf(im_sign[1] * np.imag(F.F)[fix_regions[1]] /
                               ((2 * re_sign * np.real(F.F)[fix_regions[1]])**0.5 + eps)) + 1) / 2
    factor[bad_region] *= 0

    return pd.Series(factor, index=q0.index)

##############################
#    Caustic time finding    #
##############################

CausticTimeCallback = Callable[[ArrayLike, ArrayLike, ArrayLike, bool], ArrayLike]

class CausticTimeFinderTimeTraj(TimeTrajectory):
    """
    Time trajectory class for caustic finding algorithm

    This class is in use by the function caustic_times(). It allows one
    to set the direction and step size for the propagation, allowing a safe
    propagation trajectory for the estimation of the direction for the gradient
    descent-like algorithm.

    This class expects two custom fields in its init() function
    - direction: The direction of the step in time for each trajectory. If given, \
        it is used instead of the user defined directions.
    - dt: The size of the step in time for each trajectory. If given and smaller \
        than the user defined step size, it is used instead.

    Parameters
    ----------
    dir_func : function with signature (q0, p0, t0, est) -> directions.
        Function for setting the direction of step for each trajectory.
        The function's input parameters are
            - q0: Initial positions of the trajectories, for current step
            - p0: Initial momenta of the trajectories, for current step
            - t0: Initial times of the trajectories, for current step
            - est: Whether the propagated step is used for gradient descent direction \
                estimation or not
        And the output should be
            - directions: Array of directions on the complex plane for each trajectory.

        The output directions should be the direction of step that will be taken
        when propagating in order to estimate the actual gradient descent direction.
        The directions should be complex numbers of norm 1, corresponding to unit
        vectors on the complex plane.
    dist_func : function with signature (q0, p0, t0, est) -> dists.
        Function for setting the size of step for each trajectory.
        The function's input parameters are
            - q0: Initial positions of the trajectories, for current step
            - p0: Initial momenta of the trajectories, for current step
            - t0: Initial times of the trajectories, for current step
            - est: Whether the propagated step is used for gradient descent direction \
                estimation or not
        And the output should be
            - dists: Array of step sizes on the complex plane for each trajectory.
    est : bool
        Whether the propagated step is used for gradient descent direction \
            estimation or not. Essentially this is just passed to the functions
    """

    dts: ArrayLike
    direction: ArrayLike
    path: TimeTrajectory

    def __init__(self, dir_func: CausticTimeCallback,
                 dist_func: CausticTimeCallback, est: bool):
        self.dir_func = dir_func
        self.dist_func = dist_func
        self.est = est

    def init(self, ics):
        q0, p0, t0 = ics.q0.to_numpy(), ics.p0.to_numpy(), ics.t.to_numpy()

        self.dts = self.dist_func(q0, p0, t0, self.est)

        if 'dt' in ics:
            # If outside guess is smaller than current dt, take it
            self.dts = np.min([self.dts, ics.dt.to_numpy()], axis=0)

        if 'direction' not in ics:
            self.direction = self.dir_func(q0, p0, t0, self.est)
        else:
            self.direction = np.exp(1j*ics.direction).to_numpy()

        self.path = LineTraj(0, 1, t0, t0 + self.direction * self.dts)

        return self

    def t_0(self, tau):
        return self.path.t_0(tau)

    def t_1(self, tau):
        return self.path.t_1(tau)

def caustic_times(result: FINCOResults, dir_func: CausticTimeCallback,
                  dist_func: CausticTimeCallback, n_iters: int, skip: int = -1,
                  plot_steps: bool = False, orig: Optional[FINCOResults]= None,
                  x: Optional[ArrayLike] = None, S_F: Optional[pd.Series] = None,
                  **kwargs) -> pd.Series:
    """
    Performs caustic time lookup for given trajectories in final time.

    This is done via some sort of gradient descent on abs(xi_1) of the trajectories.
    On each iteration the trajectories are propagated for on a short complex time
    interval. The resulting xi_1 values are then used to calculate an estimatied
    direction and step size for the gradient descent, which are used for the actual
    propagation.

    The algorithm also supports saving snapshots of the process as FINCO result
    files, as well as plotting of steps.

    Parameters
    ----------
    result : FINCOResults
        Results file to work with. Currently assumes final time is at timestep 1.
    dir_func : function with signature (q0, p0, t0, est) -> directions.
        Function for setting the direction of step for each trajectory.
        The function's input parameters are
            - q0: Initial positions of the trajectories, for current step
            - p0: Initial momenta of the trajectories, for current step
            - t0: Initial times of the trajectories, for current step
            - est: Whether the propagated step is used for gradient descent direction \
                estimation :math:`\\tilde\\nu`or not
        And the output should be
            - directions: Array of directions on the complex plane for each trajectory.

        The output directions should be the direction of step that will be taken
        when propagating in order to estimate the actual gradient descent direction.
        The directions should be complex numbers of norm 1, corresponding to unit
        vectors on the complex plane.
    dist_func : function with signature (q0, p0, t0, est) -> dists.
        Function for setting the size of step for each trajectory.
        The function's input parameters are
            - q0: Initial positions of the trajectories, for current step
            - p0: Initial momenta of the trajectories, for current step
            - t0: Initial times of the trajectories, for current step
            - est: Whether the propagated step is used for gradient descent direction \
                estimation or not
        And the output should be
            - dists: Array of step sizes on the complex plane for each trajectory.
    n_iters : int
        Number of iterations (steps) to run the algorithm.
    skip : int, optional
        The interval between iterations to save a snapshot of the results got so
        far, as well step plotting. Nonpositive value disables. The default is -1.
    plot_steps : bool, optional
        Whether to plot steps or not. Step plotting consists of two figures, the
        left shows which trajectories got closer to a caustic due to the last step,
        and the right showing the sign of the imaginary component of the current
        time of each trajectory. In addition, the wavefunction is reconstructed
        using the current times of each trajectory, and a plot of the reconstructed
        wavepacket from every step is shown at the end. The default is False.
    orig : FINCOResults, optional
        A FINCO results dataset containing the original final time of the trajectories.
        Can be used if one wants to continue the algorithm, and reconstruct the
        wavefunction using a different dataset. If None results is taken. The default
        is None.
    x : ArrayLike of float, optional
        x values for the reconstructed wavepacket. None indicates that wavfunction
        reconstruction should not be performed and plotted. The default is None.
    S_F : pandas.Series of complex, optional
        Calculated Berry factor resulting from Stokes treatment of the results,
        as given by calc_factor2(). If None, and plot_steps is True, a factor of
        one is used for all trajectories. The default is None.

    The rest of the parameters are passed to propagate()

    Returns
    -------
    ts: pandas.Series of complex
        The final time of each trajectory at the end of the algorithm's run.

    """

    logger = logging.getLogger('finco.caustic_times')

    if skip > 0:
        step_dir = result.file_path
        if not step_dir:
            logger.warning("""Got results from memory. Writing intermediate steps
                           to caustic_times.steps folder""")
            step_dir = 'caustic_times'

        step_dir += '.ct_steps'

        try:
            os.mkdir(step_dir)
        except FileExistsError:
            logger.warning('Directory %s already exists. Overwriting intermediate files',
                           step_dir)

        shutil.copy(result.file_path, os.path.join(step_dir, 'step_0.hdf'))

        if plot_steps:
            # Keep a view to the starting point for wavefunction reconstruction
            if orig is None:
                orig = get_view(result, 1)
            else:
                orig = get_view(orig, 1)
            n_jobs = kwargs.get('n_jobs', 1)

            if x is None:
                logger.info("""No x values were given for wavfunction recunstruction. \
                            Not reconstructing wavefunction""")

            if S_F is None:
                logger.info("No inital Berry factor was given for step plotting. \
                            Using imaginary component of time as factor")

            psis = [orig.reconstruct_psi(x, 1, S_F, n_jobs=n_jobs)]

    lr = 0.1
    est_t_traj = CausticTimeFinderTimeTraj(dir_func, dist_func, True)
    run_t_traj = CausticTimeFinderTimeTraj(dir_func, dist_func, False)

    if 'blocksize' not in kwargs:
        kwargs['blocksize'] = 2**15

    for i in range(n_iters):
        logger.info('Beginning iteration %d/%d', i+1, n_iters)
        begin = perf_counter()

        logger.debug('Running first propagation')

        kwargs.update({'trajs_path': None,
                       'time_traj': est_t_traj,
                       'drecord': 1/3})
        est = continue_propagation(result, **kwargs)

        logger.debug('Calculating propagation direction')

        xi_1p = np.stack([est.get_caustics_map(i).xi_1 for i in range(4)])
        est_dir = (est.t.loc[:,1,:] - est.t.loc[:,0,:]).to_numpy()

        del est
        xi_1_dot = ((1/3 * xi_1p[3] - 1.5 * xi_1p[2] + 3*xi_1p[1] - 11/6 * xi_1p[0]) /
                    np.abs(est_dir))
        theta = np.angle(-(xi_1p[0] / xi_1_dot / est_dir))

        logger.debug('Running second propagation')
        ics = result.get_results(1)
        ics['direction'] = theta
        ics['dt'] = np.abs(xi_1p[0]) / np.abs(xi_1_dot) * lr

        kwargs.update({'trajs_path': os.path.join(step_dir, 'last_step.hdf'),
                       'time_traj': run_t_traj,
                       'drecord': 1})
        result = propagate(ics, **kwargs)

        logger.debug('Dxi_1: %e', (result.get_caustics_map(0).xi_1.abs().sum() -
                                   result.get_caustics_map(1).xi_1.abs().sum()))

        if skip > 0 and (i + 1) % skip == 0:
            view = get_view(result, 1)

            if plot_steps:
                a = (np.abs(view.get_caustics_map(1).xi_1) -
                     np.abs(xi_1p[0])).to_numpy()
                b = np.imag(view.get_trajectories(1).t.to_numpy())
                _, [diff, times] = plt.subplots(1,2)
                diff.tripcolor(np.real(ics.q0), np.imag(ics.q0), np.sign(a))
                diff.set_xlabel('$\Re q_0$')
                diff.set_ylabel('$\Im q_0$')
                diff.set_title('$\Delta t$')
                plt.colorbar(diff.tripcolor(np.real(ics.q0), np.imag(ics.q0), np.sign(a)))
                
                times.tripcolor(np.real(ics.q0), np.imag(ics.q0), np.sign(b))
                times.set_xlabel('$\Re q_0$')
                times.set_ylabel('$\Im q_0$')
                times.set_title('Caustic time sign')
                
                plt.tight_layout()

                ts = np.imag(view.get_trajectories(1).t)
                factor = S_F*np.sign(ts) if S_F is not None else ts > 0
                psis.append(orig.reconstruct_psi(x,1,S_F=factor, n_jobs=n_jobs))

            shutil.copy(result.file_path, os.path.join(step_dir, f'step_{i+1}.hdf'))

        end = perf_counter()
        logger.info('Finished iteration %d/%d. Time: %fs',
                    i + 1, n_iters, end - begin)

    if skip > 0 and plot_steps:
        plt.figure()
        for i, psi in enumerate(psis):
            plt.plot(x,np.abs(psi),c=plt.cm.winter(i/len(psis)))

    return result.get_trajectories(1).t
