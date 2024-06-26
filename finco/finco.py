# -*- coding: utf-8 -*-
"""
The main module in FINCO. Contains an implementation of the FINCO propagation
algorithm.

The core feature of this algorithm is the ability to propagate a wavepacket
in time using semi-classical trajectories.

The algorithm supports custom trajectories in time using a TimeTrajectory object.
The algorithm propagates the system in time using a parametrized trajectory,
with parameter tau in range [0,1), and uses a provided TimeTrajectory to translate
it into position and derivative in temporal space.

For efficiency, the algorithm employs two types of parallelization, by allowing
working  in vectorized fashion on blocks of trajectories and by allowing propagation
of blocks in parallel.

In addition, the algorithm allows saving the results for analysis and wavepacket
reconstruction in a file.

All the FINCO propagation functions share the same parameters, given here:

    Required Parameters
    -------------------
    V : ArrayLike of 3 functions
        The function V(t,q) and its first two spatial derivatives. Should
        be packaged as [V, dV/dq, d^2V/dq^2]
    m : float
        System's mass.
    gamma_f : float
        Gaussian width for the Gaussians sampled for wavepacket reconstruction.
    time_traj : TimeTrajectory
        Time trajectory object for time trajectory calculation of each trajectory.
    dt : float in range of (0,1)
        Maximal step on the the parameterized time trajectoy for the propagator to take
    drecord : float in range of (0,1)
        Delta between snapshot taking time, in the time trajectory parameterization.

    Optional Parameters
    -------------------
    blocksize : int
        Block size of trajectories to process in vectorized fashion. The default is 1024.
    n_jobs : int
        Number of concurrent workers to use while propagating. Refer to
        joblib's documentation for more details. The default is 1.
    verbose : bool
        Whether to print progress information or not. The default is True.
    trajs_path : string
        Path for file to save propagated trajectories' results. If set to
        None, the results are saved in memory
        The default is 'trajs.hdf'.
    append : bool
        Whether to append data to the results file or overwrite it.
        Set to True to append. The default is False.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

from .doprop import _do_step
from .utils import hbar
from .results import FINCOResults, FINCOWriter, load_results, results_from_data
from .time_traj import TimeTrajectory

def create_ics(q0: ArrayLike, S0: ArrayLike, t0: Optional[ArrayLike] = None):
    """
    Creates a set of initial trajectory states for FINCO

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions.
    S0 : ArrayLike of 3 functions
        The function S(t=0,q) and its first two spatial derivatives. Should
        be packed as [S, dS/dq, d^2S/dq^2]
    t0 : ArrayLike of complex, optional
        Initial times for the trajectories. The default is None, on which t0=0
        is assigned to all trajectories.

    Returns
    -------
    ics : pandas.DataFrame
        DataFrame with the initial conditions. Contains the following fields

        - q0 : Initial positions. complex
        - p0 : Initial momenta. complex
        - S_20 : Initial value of second spatial derivative of S for each \
                 trajectory. complex
        - q : Current positions (same as q0). complex
        - p : Current momenta (same as p0). complex
        - t : Current (initial) time for each trajectory. complex
        - S : Current (initial) value of S for each trajectory. complex
        - Mpp, Mpq, Mqp, Mqq : Current(initial) values of the stability matrix, \
                               which is the identity matrix at t=0. complex

        and an index consisting of two fields:

        - t_index : Trajectory index
        - timestep : Timestep of entry

        This is the expected index from datasets in the library
    """
    Ss = S0[0](q0)
    ps = S0[1](q0)
    Ss_2 = S0[2](q0)
    t0 = t0 if t0 is not None else np.zeros_like(q0)

    index = pd.MultiIndex.from_product([np.arange(len(q0)), [0]],
                                       names=['t_index', 'timestep'])
    return pd.DataFrame({'q0': q0, 'p0': ps, 'S_20': Ss_2, 't': t0,
                         'q': q0, 'p': ps, 'S': Ss,
                         'Mpp': np.ones_like(q0), 'Mpq': np.zeros_like(q0),
                         'Mqp': np.zeros_like(q0), 'Mqq': np.ones_like(q0)},
                        index = index)


class FINCOConf:
    """
    Wrapper class for the configuration of FINCO.

    The class manages the default values for optional parameters, makes sure all
    required arguments were provided and no unknown arguments were passed,
    and allows simple value retrieval as attributes from the kwargs dictionary.
    """
    def __init__(self, **kwargs):
        required_args = ['V', 'm', 'gamma_f', 'time_traj', 'dt', 'drecord']
        optional_args = ['blocksize', 'n_jobs', 'trajs_path',
                         'append', 'verbose']

        # Make sure all required args are there
        for arg in required_args:
            if arg not in kwargs:
                raise ValueError(f'Configuration missing required argument {arg}')

        # Make sure no unknown args are there
        for arg in kwargs:
            if arg not in required_args + optional_args:
                raise ValueError(f'Configuration got unknown argument {arg}')


        self._args = {'blocksize': 1024,
                      'n_jobs': 1,
                      'trajs_path': 'trajs.hdf',
                      'append': False,
                      'verbose': True}
        self._args.update(kwargs)

    def __repr__(self):
        """
        Return the canonical string representation of the object.
        In this case, the dictionary's representation is returned
        """
        return repr(self._args)

    def __getattr__(self, name):
        """Convenience value retrieval from the dictionary as attribute"""
        return self._args[name]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


def calc_xi_1(sol, S_20, t_0: float, t_1: float, gamma_f: float,
              ref_angle: Optional[ArrayLike] = None) -> [ArrayLike, ArrayLike,
                                                         ArrayLike, ArrayLike]:
    """
    Calculates the norm and angle of xi_1 over the time parameter range
    (t0, t1) using a functional solution, assuring a smooth angle.

    Parameters
    ----------
    sol : Solution functional
        A solution function that takes a time parameter and returns the
        trajectories' parameters at that time.
    S_20 : ArrayLike
        Array of the initial values of the second spatial derivative of S
    t_0 : float in [0, 1]
        Initial time parameter.
    t_1 : float in [0, 1]
        Final time parameter.
    gamma_f : float
        Gaussian width for the Gaussians sampled for wavepacket reconstruction.
    ref_angle : ArrayLike of float, optional
        Reference initial angles to work with. The default is None.

    Returns
    -------
    xi_1_abs_0: ArrayLike of float
        The norm of xi_1 at t0.
    xi_1_abs_1: ArrayLike of float
        The norm of xi_1 at t1.
    xi_1_angle_0: ArrayLike of float
        The angle of xi_1 at t0.
    xi_1_angle_1: ArrayLike of float
        The angle of xi_1 at t1.
    """
    res = np.array([sol(t) for t in np.linspace(t_0, t_1, 50)]).T.reshape((7,-1,50))
    Mpp, Mpq, Mqp, Mqq = res[3:]
    xi_1 = (2 * gamma_f * (Mqq + S_20[:,np.newaxis] * Mpq) -
            1j / hbar * (Mqp + S_20[:,np.newaxis] * Mpp))

    xi_1_angle = np.angle(xi_1)
    if ref_angle is not None:
        xi_1_angle[:,0] = ref_angle
    xi_1_abs = np.abs(xi_1)
    xi_1_angle = np.unwrap(xi_1_angle/2)*2

    return xi_1_abs[:,0], xi_1_abs[:,-1], xi_1_angle[:,0], xi_1_angle[:,-1]

def calc_xis(sol, S_20, Ts: ArrayLike, gamma_f: float,
             ref_angle: Optional[ArrayLike] = None) -> [ArrayLike, ArrayLike]:
    """
    Calculates the norm and angle of xi_1 at sampled times Ts using a
    functional solution, assuring a smooth angle.

    Parameters
    ----------
    sol : Solution functional
        A solution function that takes a time parameter and returns the
        trajectories' parameters at that time.
    S_20 : ArrayLike
        Array of the initial values of the second spatial derivative of S
    Ts : ArrayLike of floats in range [0, 1]
        Times to sample xi_1 at. Expected to be sorted in ascending way.
    gamma_f : float
        Gaussian width for the Gaussians sampled for wavepacket reconstruction.
    ref_angle : ArrayLike of float, optional
        Reference initial angles to work with. The default is None.

    Returns
    -------
    xi_1_abs: ArrayLike of float
        The norm of xi_1 at the sampled times.
    xi_1_angle: ArrayLike of float
        The angle of xi_1 at the sampled times.
    """
    xi_1_angle = [None] * len(Ts)
    xi_1_abs = [None] * len(Ts)

    xi_1_abs[0], _, xi_1_angle[0], _ = calc_xi_1(sol, S_20, Ts[0], Ts[1], gamma_f)
    if ref_angle is not None:
        xi_1_angle[0] = ref_angle

    for i in range(len(Ts) - 1):
        _, xi_1_abs[i+1], _, xi_1_angle[i+1] = calc_xi_1(sol, S_20, Ts[i], Ts[i+1],
                                                         gamma_f, ref_angle)
        ref_angle = xi_1_angle[i+1]

    return np.stack(xi_1_abs).T, np.stack(xi_1_angle).T

def propagate_traj(ics: pd.DataFrame, V: ArrayLike, m: float,
                   time_traj: TimeTrajectory, max_step: float, Ts: ArrayLike) -> ArrayLike:
    """
    Propagates a block of intial states in time in the system. The function
    does the propagation and returns the results.

    Used as a utility function. Users should use the 'propagate' function.

    Parameters
    ----------
    ics : pandas.DataFrame
        Initial trajectory states dataset. Should contain at least the fields
        created by create_ics()
    V : ArrayLike of 3 functions
        The function V(t,q) and its first two spatial derivatives. Should
        be packaged as [V, dV/dq, d^2V/dq^2]
    m : float
        System's mass.
    time_traj : TimeTrajectory
        Time trajectory object for time trajectory calculation of each trajectory.
    max_step : float in range of (0,1)
        Maximal step on the the parameterized time trajectoy for the propagator
        to take.
    Ts : ArrayLike of floats in range of (0,1)
        Increasing sequence of times to take a trajectory snapshot, in the
        time trajectory parameterization.

    Returns
    -------
    results : ArrayLike
        The propagation results, as an array of (8*n_trajectories, n_timesteps)
        Where the columns are:
            - t: The trajectory's time at the timestep
            - q: The trajectory's position at the timestep
            - p: The trajectory's momentum at the timestep
            - S: The trajectory's action at the timestep
            - M_pp, M_pq, M_qp, M_qq : The stability matrix elements
    """
    # Prepare for propagation and propagate
    # TODO: Reinroduce angle unwrapping and calculation here and in result appending below.

    # Parameter order: q, p, S, M_pp, M_pq, M_qp, M_qq
    y0 = np.concatenate([ics.q, ics.p, ics.S, ics.Mpp, ics.Mpq, ics.Mqp, ics.Mqq])

    # ref_angle = np.angle(2 * gamma_f - 1j / hbar * ics.S_20)
    time_traj.init(ics)
    discont_times = time_traj.get_discontinuity_times()
    results = []

    for t0, t1 in zip([0] + discont_times, discont_times + [1]):
        t_eval = Ts[Ts <= t1]
        if t0 > 0:
            t_eval = t_eval[t_eval > t0]

        res = solve_ivp(_do_step, (t0, t1),
                        y0, args=(time_traj.t_funcs, V, m),
                        t_eval=None if len(t_eval) == 0 else t_eval, max_step=max_step,
                        vectorized=True, dense_output=True)

        if res.status != 0:
            return None

        # xi_1_abs, xi_1_angle = calc_xis(res.sol, ics.S_20.to_numpy(), [t0] + list(t_eval) + [t1],
        #                         gamma_f, ref_angle)
        # ref_angle = xi_1_angle[:,-1]

        y0 = res.sol(t1)
        if len(t_eval) > 0:
            result = res.y.reshape(7, -1, len(res.t))
            t = np.array([time_traj.t_0(tau) for tau in res.t]).T[np.newaxis,:]
            # result[3], result[4] = xi_1_abs[:,1:-1], xi_1_angle[:,1:-1]
            result = np.concatenate((t, result))
            results.append(result)

    return np.concatenate(results, axis=2).reshape(8, -1)

def propagate(ics: pd.DataFrame, **kwargs) -> FINCOResults:
    """
    Propagates the system in time, given a set of initial conditions.

    Parameter list is specified at the module documentation

    Parameters
    ----------
    ics : pandas.DataFrame
        Initial trajectory states dataset. Should contain at least the fields
        created by create_ics()

    Returns
    -------
    results : FINCOReader
        The propagation results.
    """
    def split_df(df, nblocks):
        idxs = np.arange(len(df))
        return [df.iloc[blk] for blk in np.array_split(idxs, nblocks)]
        
    def process_block(block):
        return propagate_traj(block, conf.V, conf.m, time_traj=conf.time_traj,
                              max_step=conf.dt, Ts=Ts)

    logger = logging.getLogger('finco.propagation')
    conf = FINCOConf(**kwargs)
    Ts = np.arange(0, 1+conf.drecord/2, conf.drecord)

    #prepare slices
    n_jobs = conf.n_jobs if conf.n_jobs > 0 else cpu_count() + 1 + conf.n_jobs
    nslices = int(np.ceil(ics.q.size / conf.blocksize / n_jobs))
    slices = split_df(ics, nslices)

    logger.debug("Starting propagation of %d trajectories in %d slices",
                len(ics), nslices)

    with FINCOWriter(file_path=conf.trajs_path, append=conf.append) as file:
        with Parallel(n_jobs=n_jobs) as parallel:
            for slc in tqdm(slices, disable=not conf.verbose):
                #prepare blocks
                nblocks = int(np.ceil(slc.shape[0] / conf.blocksize))
                blocks = split_df(slc, nblocks)

                results = parallel(delayed(process_block)(block)
                                   for block in blocks)

                for block, result in zip(blocks, results):
                    if result is None:
                        continue

                    q0, p0, S_20 = block.q0, block.p0, block.S_20
                    t, q, p, S, Mpp, Mpq, Mqp, Mqq = result

                    q0 = np.tile(q0, (Ts.size, 1)).T.flatten()
                    p0 = np.tile(p0, (Ts.size, 1)).T.flatten()
                    S_20 = np.tile(S_20, (Ts.size, 1)).T.flatten()

                    index = pd.MultiIndex.from_product([block.index.get_level_values(0),
                                                        np.arange(Ts.size)],
                                                       names=['t_index','timestep'])

                    res = pd.DataFrame({'q0': q0, 'p0': p0, 'S_20': S_20, 't': t,
                                         'q': q, 'p': p, 'S': S,
                                         'Mpp': Mpp, 'Mpq': Mpq, 'Mqp': Mqp, 'Mqq': Mqq},
                                        index=index)

                    file.add_results(res)

    if conf.trajs_path is not None:
        return load_results(conf.trajs_path, conf.gamma_f)

    return results_from_data(file.data, conf.gamma_f)

def continue_propagation(results: FINCOResults, **kwargs) -> FINCOResults:
    """
    Convenience function. Continues a propagation of system in time, given a
    FINCO results object, by using its last timestep as initial trajectory state.

    Parameters
    ----------
    results : FINCOResults
        A FINCO results object holding the trajectories' data, for further propagation.

    The rest of the parameters are passed to propagate()

    Returns
    -------
    results : FINCOReader
        The propagation results.
    """
    res = results.get_results()
    last_time = np.max(res.index.get_level_values('timestep'))
    ics = res.loc[:, last_time, :]
    ics = ics.set_index(pd.MultiIndex.from_product([ics.index, [last_time]],
                                                   names=['t_index','timestep']))

    return propagate(ics, **kwargs)
