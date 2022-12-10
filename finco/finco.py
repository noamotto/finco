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
    heuristics : ArrayLike of Heuristic objects
        List of Heuristics to apply on each propagated trajectory, in order
        to throw invalid trajectories.
        The default is None.
    blocksize : int
        Number of trajectories to process in parallel. The default is 1024.
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

def create_ics(q0, S0, gamma_f = 1, t0 = None):
    """
    Creates a set of initial trajectory states for FINCO

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions.
    S0 : ArrayLike of 3 functions
        The function S(t=0,q) and its first two spatial derivatives. Should
        be packed as [S, dS/dq, d^2S/dq^2]
    gamma_f : float, optional
        Gaussian width for the Gaussians sampled for wavepacket reconstruction.
        The default is 1.
    t0 : ArrayLike of complex, optional
        Initial times for the trajectories. The default is None, on which t0=0
        is assigned to all trajectories.

    Returns
    -------
    ics : pandas.DataFrame
        DataFrame with the initial conditions. Contains the following fields

        - q0 : Initial positions. complex
        - p0 : Initial momenta. complex
        - q : Current positions (same as q0). complex
        - p : Current momenta (same as p0). complex
        - t : Current (initial) time for each trajectory. complex
        - S : Current (initial) value of S for each trajectory. complex
        - S_2 : Current (initial) value of second spatial derivative of S for \
                each trajectory. complex
        - xi_1_abs : Current (initial) absolute value of the first derivative of xi w.r.t q0
        - xi_1_abs : Current (initial) phase of the first derivative of xi w.r.t q0

        and an index consisting of two fields:

        - t_index : Trajectory index
        - timestep : Timestep of entry

        This is the expected index from diatasets in the library
    """
    Ss = S0[0](q0)
    ps = S0[1](q0)
    Ss_2 = S0[2](q0)
    xi_1 = 2*gamma_f - 1j / hbar * Ss_2
    t0 = t0 if t0 is not None else np.zeros_like(q0)

    index = pd.MultiIndex.from_product([np.arange(len(q0)), [0]],
                                       names=['t_index', 'timestep'])
    return pd.DataFrame({'q0': q0, 'p0': ps, 'q': q0,
                         'p': ps, 't': t0, 'S': Ss, 'S_2': Ss_2,
                         'xi_1_abs': np.abs(xi_1), 'xi_1_angle': np.angle(xi_1)},
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
        optional_args = ['heuristics', 'blocksize', 'n_jobs', 'trajs_path',
                         'append', 'verbose']
        
        # Make sure all required args are there
        for arg in required_args:
            if arg not in kwargs:
                raise ValueError('Configuration missing required argument {}'.format(arg))
                
        # Make sure no unknown args are there
        for arg in kwargs:
            if arg not in required_args + optional_args:
                raise ValueError('Configuration got unknown argument {}'.format(arg))
        
        
        self._args = {'blocksize': 1024,
                      'n_jobs': 1,
                      'trajs_path': 'trajs.hdf',
                      'append': False,
                      'verbose': True,
                      'heuristics': None}
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
    

def calc_xi_1(sol, t_0, t_1, gamma_f, ref_angle=None):
    """
    Calculates the norm and angle of xi_1 over the time parameter range
    (t0, t1) using a functional solution, assuring a smooth angle.

    Parameters
    ----------
    sol : Solution functional
        A solution function that takes a time parameter and returns the
        trajectories' parameters at that time.
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
    res = np.array([sol(t) for t in np.linspace(t_0, t_1, 50)]).T.reshape((5,-1,50))
    Mp, Mq = res[3:]
    xi_1 = (2 * gamma_f * Mq - 1j / hbar * Mp)

    xi_1_angle = np.angle(xi_1)
    if ref_angle is not None:
        xi_1_angle[:,0] = ref_angle
    xi_1_abs = np.abs(xi_1)
    xi_1_angle = np.unwrap(xi_1_angle/2)*2

    return xi_1_abs[:,0], xi_1_abs[:,-1], xi_1_angle[:,0], xi_1_angle[:,-1]

def calc_xis(sol, Ts, gamma_f, ref_angle=None):
    """
    Calculates the norm and angle of xi_1 at sampled times Ts using a
    functional solution, assuring a smooth angle.

    Parameters
    ----------
    sol : Solution functional
        A solution function that takes a time parameter and returns the
        trajectories' parameters at that time.
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

    xi_1_abs[0], _, xi_1_angle[0], _ = calc_xi_1(sol, Ts[0], Ts[1], gamma_f)
    if ref_angle is not None:
        xi_1_angle[0] = ref_angle

    for i in range(len(Ts) - 1):
        _, xi_1_abs[i+1], _, xi_1_angle[i+1] = calc_xi_1(sol, Ts[i], Ts[i+1], gamma_f,
                                                ref_angle)
        ref_angle = xi_1_angle[i+1]

    return np.stack(xi_1_abs).T, np.stack(xi_1_angle).T
    
def propagate_traj(ics, V, m, time_traj: TimeTrajectory, max_step, Ts, gamma_f):
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
        The propagation results, as an array of (7*n_trajectories, n_timesteps)
        Where the columns are:
            - t: The trajectory's time at the timestep
            - q: The trajectory's position at the timestep
            - p: The trajectory's momentum at the timestep
            - S: The trajectory's action at the timestep
            - xi_1_abs: The trajectory's norm of xi_1 at the timestep
            - xi_1_angle: The trajectory's phase of xi_1 at the timestep
            - S_2: The trajectory's action's second derivative at the timestep
    """
    # Prepare for propagation and propagate
    # Calc M_p, M_q from S0_2, xi_1
    xi_1_0 = ics.xi_1_abs * np.exp(1j * ics.xi_1_angle)
    M = np.array([[ics.S_2, -np.ones_like(ics.q)],
                  [np.full_like(ics.q,2*gamma_f), np.full_like(ics.q,-1j/hbar)]])
    M_q, M_p = np.einsum('ijn,jn->in', np.linalg.pinv(M.T).T, [np.zeros_like(ics.q), xi_1_0])

    # Parameter order: q, p, S, M_p, M_q
    y0 = np.array([ics.q, ics.p, ics.S, M_p, M_q]).flatten()

    ref_angle = ics.xi_1_angle
    time_traj.init(ics)
    discont_times = time_traj.get_discontinuity_times()
    results = []

    for t0, t1 in zip([0] + discont_times, discont_times + [1]):
        t_eval = Ts[Ts <= t1]
        if t0 > 0:
            t_eval = t_eval[t_eval > t0]

        res = solve_ivp(_do_step, (t0, t1),
                        y0, args=(time_traj, V, m),
                        t_eval=None if len(t_eval) == 0 else t_eval, max_step=max_step,
                        vectorized=True, dense_output=True)

        if res.status != 0:
            return None

        xi_1_abs, xi_1_angle = calc_xis(res.sol, [t0] + list(t_eval) + [t1],
                               gamma_f, ref_angle)
        ref_angle = xi_1_angle[:,-1]

        y0 = res.sol(t1)
        if len(t_eval) > 0:
            result = res.y.reshape(5, -1, len(res.t))
            S_2 = (result[3] / result[4])[np.newaxis,:]
            t = np.array([time_traj.t_0(tau) for tau in res.t]).T[np.newaxis,:]
            result[3], result[4] = xi_1_abs[:,1:-1], xi_1_angle[:,1:-1]
            result = np.concatenate((t, result, S_2))
            results.append(result)

    return np.concatenate(results, axis=2).reshape(7, -1)

def propagate(ics, **kwargs) -> FINCOResults:
    """
    Propagates the system in time, given a set of initial conditions.
    Currently assumes the propagation starts at t=0.
    
    Parameter list is specified at the module documentation

    Returns
    -------
    results : FINCOReader
        The propagation results.
    """
    def process_block(block):
        return propagate_traj(block, conf.V, conf.m, time_traj=conf.time_traj,
                              max_step=conf.dt, Ts=Ts, gamma_f=conf.gamma_f)

    conf = FINCOConf(**kwargs)
    Ts = np.arange(0, 1+conf.drecord/2, conf.drecord)

    #prepare slices
    n_jobs = conf.n_jobs if conf.n_jobs > 0 else cpu_count() + 1 + conf.n_jobs
    nslices = int(np.ceil(ics.q.size / conf.blocksize / n_jobs))
    slices = np.array_split(ics, nslices)

    with FINCOWriter(file_path=conf.trajs_path, append=conf.append) as file:
        with Parallel(n_jobs=n_jobs) as parallel:
            for slc in tqdm(slices, disable=not conf.verbose):
                #prepare blocks
                nblocks = int(np.ceil(slc.shape[0] / conf.blocksize))
                blocks = np.array_split(slc, nblocks)

                results = parallel(delayed(process_block)(block)
                                   for block in blocks)

                for block, result in zip(blocks, results):
                    if result is None:
                        continue

                    q0, p0 = block.q0, block.p0
                    t, q, p, S, xi_1_abs, xi_1_angle, S_2 = result

                    q0 = np.tile(q0, (Ts.size, 1)).T.flatten()
                    p0 = np.tile(p0, (Ts.size, 1)).T.flatten()

                    index = pd.MultiIndex.from_product([block.index.get_level_values(0),
                                                        np.arange(Ts.size)],
                                                       names=['t_index','timestep'])

                    res = pd.DataFrame({'q0': q0, 'p0': p0,
                                        't': t, 'q': q, 'p': p, 'S': S, 'S_2': S_2,
                                        'xi_1_abs': xi_1_abs, 'xi_1_angle': xi_1_angle},
                                        index=index)

                    file.add_results(res)

        if file.results_file is not None:
            return load_results(conf.trajs_path, conf.gamma_f)

        return results_from_data(file.data, conf.gamma_f)

def continue_propagation(results, **kwargs) -> FINCOResults:
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
