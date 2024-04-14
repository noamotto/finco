# -*- coding: utf-8 -*-
"""
Configuration file for propagating the 1D Coulomb ground state using FINCO. It
contains the system's parameters and functions for the potential and initial
state, as well as a class for mapping between initial conditions and trajectories
in time and functions for dealing with nonphysical contributions.

Remarks
-------

1. The coordinate q here is radial, hence the lacking of absolute values.
2. The usage of the functions dealing with nonphysical contributions can be seen
in various exploration and analysis scripts in this folder. Refer to `caustic_times.py`
for usage of finding caustic times, and to `analyze_results.py` for an example of
using locate_caustics() and eliminate_stokes().
"""

#%% Setup

import os
import logging

import numpy as np
import pandas as pd
from joblib import cpu_count

from finco.time_traj import SequentialTraj, CircleTraj, LineTraj
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, approximate_F

# Logging
logging.basicConfig()

# Determine default number of jobs, as number of physical cores - 1 or
# using a designated environment variable
n_jobs = int(os.getenv('NCPUS', default = cpu_count(True) - 1))

# System params
m = 1
q_e = 1

halfcycle = 2 * np.pi

def S0_0(q):
    return 1j * (q - np.log(q) - 0.5*np.log(2))

def S0_1(q):
    return np.array(1j * (1 - 1./q))

def S0_2(q):
    return 1j / q**2

def V_0(q, _):
    return -q_e / q

def V_1(q, _):
    return q_e / q**2

def V_2(q, _):
    return -2 * q_e / q**3

V = [V_0, V_1, V_2]
S0 = [S0_0, S0_1, S0_2]

def sign(a):
    """
    Sign function treating 0 as negative

    Parameters
    ----------
    a : ArrayLike
        Input array.

    Returns
    -------
    b : ArrayLike
        Signs array, where b[i] = 1 if a[i] > 0 otherwise b[i] = -1
    """
    return (a > 0).astype(int) * 2 - 1

def coulombg_pole(q0, p0, n=0):
    """
    Analytically calculates the temporal position of the poles in the potential,
    given a set of initial conditions

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions to calculate for
    p0 : ArrayLike of complex of same length as q0
        Initial momenta to calculate for
    n : integer or ArrayLike of integers, optional
        Pole indices to calculate. Note that if the length of q0 is 1 the array
        of n will calculate different pole indices for the same initial condition.
        If the length of q0 is bigger than 1 then it is expecte of n to be either
        an integer indicating the same pole index to calculate for all initial
        conditions, of the same length as q0 indicating what pole index to calculate
        for each initial condition. Calculating multiple pole indices for multiple
        initial conditions is currently unsupported. The default is 0.

    Returns
    -------
    poles : ArrayLike of complex
        The calculated poles.
    """
    E0 = p0**2/2/m - 1/q0
    s = sign(q0.real)*sign(q0.imag)
    return (-q0*p0/2/E0 + (m/2)**0.5 * (np.log((2*E0/m)**0.5*s*p0*q0 + 2*E0*q0 + 1)
                                        + n*1j*np.pi*2)/2/E0**1.5/s)

def coulombg_diff(q0, p0):
    """
    Calculates the difference between two poles on the ladder for given initial
    conditions.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions to calculate for
    p0 : ArrayLike of complex of same length as q0
        Initial momenta to calculate for

    Returns
    -------
    diffs : ArrayLike of complex
        The calculated differences.
    """
    return coulombg_pole(q0, p0, n=1) - coulombg_pole(q0, p0, n=0)


class CoulombGTimeTrajectory(SequentialTraj):
    """
    Time trajectory for Coulomb ground state with no external field. Uses the
    analytical calculation of the pole ladders to create a trajectory circling
    poles on ladder, with optional additional circlings around the last pole.

    Parameters
    ----------
    n : integer
        Number of poles on the ladder to circle for each initial condition.
    k : integer, optional
        Number of additional times to circle the last pole. The default is 0.
    t : float or Callable of form (ArrayLike, ArrayLike) -> ArrayLike, optional
        Final propagation time. If the input is float then the final time of all
        trajectories is set to be the same. If not, then t is called with the
        initial conditions to determine the final time for each initial condition.
        The default is 3*2*halfcycle.
    """

    def __init__(self, n, k = 0, t = 3*2*halfcycle):
        super().__init__(t0=0, t1=1)

        self.n = n
        self.k = k

        if isinstance(t, float):
            self.t = lambda q,p: np.full_like(q, t)
        else:
            self.t = t

    def init(self, ics):
        # Calc trajectory parameters
        q0, p0, t0 = ics.q0.to_numpy(), ics.p0.to_numpy(), ics.t.to_numpy()

        # Calc the 2 directions for the trajectory
        # r: in the direction of the poles line, towards positive time
        #   with length of the circle's radius
        # u: perpendicular to r, with the same length, in the direction
        #   pointing to the poles line from the origin
        self.r = np.array(-coulombg_diff(q0, p0)) / 2
        self.r *= np.sign(self.r.real)
        self.u = (coulombg_pole(q0, p0, n=0) -
                  (np.real(coulombg_pole(q0, p0, n=0)*self.r.conj()) * self.r /
                   np.abs(self.r)**2))
        self.u *= np.abs(self.r) / np.abs(self.u)

        # Calc whether u points "above" the poles line or "below". Needed to
        # calculate the circle's entry point and to calculate which trajectories
        # need additional rotation.
        self.dir = np.sign((self.u*self.r.conj()).imag)
        self.first = np.zeros(q0.shape)
        # self.first[(q0.real < 0) & (self.dir < 0)] -= 1


        # Calc points of reference on the trajectory

        # t0: Initial point
        t0 = t0 if t0 is not None else np.zeros_like(q0)

        # t1: Final point
        t1 = self.t(q0, p0)

        # a: Point starting to circle poles
        self.nfirst = np.zeros(q0.shape)
        self.nfirst[(q0.imag < 0) & (self.dir > 0)] -= 1
        self.a = coulombg_pole(q0, p0, n=self.nfirst) - self.u
        self.turns = np.ones(q0.shape)
        self.outer = np.zeros(q0.shape)
        self.outer[(q0.real < 0) & (self.dir < 0)] -= 0.5
        # self.a[q0.imag < 0] -= self.r *2

        # b: Point of exit from the poles line
        self.b = self.a + ((self.n - 1) * 2)*self.r

        # Build path
        self.path = []
        self.discont_times = []
        if self.n==0:
            self.discont_times = [1/2]

            self.path.append(LineTraj(t0=0, t1=1/2, a=t0, b=self.a))
            self.path.append(LineTraj(t0=1/2, t1=1, a=self.a, b=t1))

        elif self.n==1:
            self.discont_times = [1/3, 2/3]

            self.path.append(LineTraj(t0=0, t1=1/3, a=t0, b=self.a))
            self.path.append(CircleTraj(t0=1/3, t1=2/3, a=self.a, r=self.r,
                                        turns=self.turns + self.k, phi0=-np.pi/2*self.dir))
            self.path.append(LineTraj(t0=2/3, t1=1, a = self.a, b=t1))

        else:
            Ts = list(np.linspace(1/(2*self.n+1), 1-1/(2*self.n+1), 2*self.n))
            self.discont_times = Ts

            da = np.zeros_like(q0)
            da[(q0.real < 0) & (self.dir < 0)] += 2 * self.u[(q0.real < 0) & (self.dir < 0)]
            b = self.a + da + 2*self.r
            self.path.append(LineTraj(t0=0, t1=Ts[0], a=t0, b=self.a))
            self.path.append(CircleTraj(t0=Ts[0], t1=Ts[1], a=self.a, r=self.r,
                                        turns=self.turns + self.outer, phi0=-np.pi/2*self.dir))
            self.path.append(LineTraj(t0=Ts[1], t1=Ts[2],
                                      a=self.a + da, b=b))

            for i in range(self.n-2):
                a = self.a + 2*(i+1)*self.r
                a[(q0.real < 0) & (self.dir < 0)] += 2 * self.u[(q0.real < 0) & (self.dir < 0)]
                phi0 = -np.pi/2*self.dir
                phi0[(q0.real < 0) & (self.dir < 0)] *= -1

                self.path.append(CircleTraj(t0=Ts[2*i+2], t1=Ts[2*i+3],
                                            a=a, r=self.r, turns=self.turns, phi0=phi0))
                self.path.append(LineTraj(t0=Ts[2*i+3], t1=Ts[2*i+4], a=a, b=a + 2*self.r))

            b = self.b + da
            phi0 = -np.pi/2*self.dir
            phi0[(q0.real < 0) & (self.dir < 0)] *= -1
            self.path.append(CircleTraj(t0=Ts[-2], t1=Ts[-1],
                                        a=b, r=self.r,
                                        turns=self.turns + self.outer + self.k, phi0=phi0))
            self.path.append(LineTraj(t0=Ts[-1], t1=1, a=self.b, b=t1))

        return self

def locate_caustics(result, n, t, n_jobs=1):
    """
    Locates caustics in results dataset.

    The algorithm begins by locating the points with lowest abs(xi_1), indicating
    a caustic nearby, then extracting a candidate by separating the points into
    blobs of neighboring points, and finally running root search on each candidate.

    Parameters
    ----------
    result : FINCOResults
        Results data to calculate for.
    n : integer
        Number of poles circled in the dataset. Used in the root search to propagate
        the trajectories correctly.
    t : float
        Final propagation time in the dataset. Used in the root search to propagate
        the trajectories correctly.
    n_jobs : positive integer, optional
        Number of concurrent workers to use in the root search. The default is 1.

    Returns
    -------
    caustics : pandas.Series
        Series of found caustics, as returned by find_caustics()
    """
    logger = logging.getLogger('analysis.stokes')

    deriv = result.get_caustics_map(1)

    blobs = separate_to_blobs(deriv, quantile=2e-2, connectivity=2)
    qs = [deriv.q0[deriv.xi_1.abs()[list(blob)].idxmin()] for blob in blobs]

    logger.info('Found %d caustic candidates', len(qs))
    logger.debug('located caustic candidates at\n %s', qs)
    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2],
                             time_traj=CoulombGTimeTrajectory(n, t=t), dt=1e-4, gamma_f=1,
                             n_jobs=n_jobs)

    caustics = caustics[np.real(caustics.q) > 0]

    logger.info('Caustic root search yielded %d caustics', len(caustics))
    logger.debug('located caustics at\n %s', caustics.q.to_numpy())

    return caustics

def eliminate_stokes(result, caustics, sigma=None):
    """
    Eliminates nonphysical contribution regions in dataset, using a list of
    found caustics.

    Parameters
    ----------
    result : FINCOResults
        Results data to treat.
    caustics : pandas.Series
        Series of found caustics, as returned by find_caustics()
    sigma : pandas.Series, optional
        Alternative values of sigma to use for the treatment. If None, the values
        in the results dataset are used. The default is None.

    Returns
    -------
    S_F : pandas.Series
        List of factors in [0,1] for each trajectory, resulting from the Stokes
        treatment.

    """
    logger = logging.getLogger('analysis.stokes')

    # Filter the small caustics. 50 Works well here as a magic number...
    # caustics = caustics[np.abs(caustics.xi_2) < 50]

    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)

    if sigma is None:
        sigma = proj.sigma

    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (_, caustic) in caustics.iterrows():
        logger.debug('handling caustic at %f', caustic.q)
        s_f = calc_factor2(caustic, proj.q0, proj.xi, sigma)
        F, _ = approximate_F(proj.q0, proj.xi, caustic)
        r = np.abs(-caustic.xi_2*2/caustic.xi_3)
        s_f[np.abs(F.v_t) > r] = 1
        S_F *= s_f
    S_F *= (np.real(proj.sigma) <= 0)
    S_F *= (np.abs(deriv.xi_1) <= 100)

    return S_F

def coulombg_caustic_times_dist(q0, p0, _, est):
    """
    Setting the step size for each trajectory for the current caustic finding
    algorithm step, based on the parameters before the step. Used both in the
    gradient descent esitmation and the actual propagation, as an estimation of
    valid step size.

    Calculation is done here based on the distance between two poles, to prevent
    accidentaly going through a pole resulting with numerical errors.

    The function's input parameters are
        - q0: Initial positions of the trajectories, for current step
        - p0: Initial momenta of the trajectories, for current step
        - t0: Initial times of the trajectories, for current step
        - est: Whether the propagated step is used for gradient descent direction \
            estimation or not
    And the output should be
        - dists: Array of step sizes on the complex plane for each trajectory.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions of the trajectories, for current step
    p0 : ArrayLike of complex
        Initial momenta of the trajectories, for current step
    t0 : ArrayLike of complex
        Initial times of the trajectories, for current step
    est : Whether the propagated step is used for gradient descent direction \
        estimation :math:`\\tilde\\nu`or not

    Returns
    -------
    dts : ArrayLike of float
        Step sizes on the for each trajectory for the current step.
    """
    if est:
        dt = 1e-2
    else:
        dt = 1e-2

    r = np.array(-coulombg_diff(np.array(q0), np.array(p0)) / 2)
    dts = np.abs(r) * dt
    dts[np.abs(r) > 1] = dt
    return dts

def coulombg_caustic_times_dir(q0 ,p0 ,t0, _):
    """
    Chooses the propagation direction for the current caustic finding algorithm
    step, based on the parameters before the step, to be used for gradient descent
    estimation for xi_1.

    Calculation is done here based on heuristic of going tangently to the closest
    pole on the ladder, to prevent numerical errors.

    Parameters
    ----------
    q0 : ArrayLike of complex
        Initial positions of the trajectories, for current step
    p0 : ArrayLike of complex
        Initial momenta of the trajectories, for current step
    t0 : ArrayLike of complex
        Initial times of the trajectories, for current step
    est : Whether the propagated step is used for gradient descent direction \
        estimation :math:`\\tilde\\nu`or not

    Returns
    -------
    directions: ArrayLike of complex
        Directions on the complex plane for each trajectory for the current step.
    """
    r = np.array(-coulombg_diff(np.array(q0), np.array(p0)) / 2)
    direction = np.array([t0 - r * 1j, t0 + r * 1j])
    close = np.argmin(np.abs(direction - t0)[:,np.newaxis], axis=0)
    direction = np.take_along_axis(direction, close, axis=0).squeeze()
    return direction / np.abs(direction)
