# -*- coding: utf-8 -*-
"""
Definition of FINCO's time trajectory and additional utility time trajectories.

In addition to the trajectory baseclass this file contains several utility
trajectory classes:

    - Trajectory building utilities:
        1. LineTraj - follows a trajectory in straight line
        2. CircleTrajec - follows a trajectory along a circle
    - Trajecty utility baseclasses
        1. SequentialTraj - allows convenient stacking of trajectories in
        sequential order.
        2. OdeSolTraj - trajectory following an ODE solution object

@author: Noam Ottolenghi
"""

from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

class TimeTrajectory:
    """
    Baseclass for time trajectories.

    The object should expose functions for t(tau) and dt/dtau(tau), for the
    FINCO algorithm to use, one for each propagated trajectory.
    It should also expose an initialization function in order to initialize the
    trajectories given their initial conditions.

    It should also expose a list of breaking points in the trajectory as
    numbers in (0,1). For a smooth trajectory an empty list should be provided
    """
    def init(self, ics: pd.DataFrame):
        """
        Initializes the time trajectories given a set of initial conditions.

        Parameters
        ----------
        ics : pandas.DataFrame
            Trajectories' initial conditions. Should contain the following fields

                - q: Initial positions
                - p: Initial momenta
                - t: Initial times for each trajectory
            Can contain additional fields.

        Returns
        -------
            self, for chaining.
        """
        return self

    def t_0(self, tau: float) -> ArrayLike:
        """
        Returns the value of t(tau) for each trajectory

        Parameters
        ----------
        tau : float in range [0,1)
            trajectory parameter.

        Returns
        -------
        t : ArrayLike of complex
            The times t(tau) for each trajectory.

        """
        raise NotImplementedError

    def t_1(self, tau: float) -> ArrayLike:
        """
        Returns the value of dt/dtau(tau) (first derivative of t w.r.t. tau)
        for each trajectory

        Parameters
        ----------
        tau : float in range [0,1)
            trajectory parameter.

        Returns
        -------
        dt/dtau : ArrayLike of complex
            The first derivatives of t w.r.t. tau for each trajectory.

        """
        raise NotImplementedError

    @property
    def t_funcs(self):
        """
        Convenience property returning both t_0 and t_1 as list
        """
        return [self.t_0, self.t_1]

    def get_discontinuity_times(self) -> List[float]:
        """
        Retrieves a list of breaking times in the trajectory, meaning points
        where the ttrajectory is not smooth. Should be called after the
        trajectories are initialized.

        Returns
        -------
        discont_times : list
            The list of breaking times in the trajectory, meaning points
            where the ttrajectory is not smooth.
        """
        return []

    def length(self):
        """
        Returns the length of the trajectory.
        """
        return 0

#######################################
#        Utility trajectories         #
#######################################

class CircleTraj(TimeTrajectory):
    """
    Follows circle trajectories in time. It follows the following formula:

    .. math::
         t(\\tau) = a - r \\exp(i \\phi_{0}) +
             r\\exp\\left(2i\\pi N_{turns} \\frac{\\tau - t_0}{t_1 - t_0} + i\\phi_0\\right)

    Parameters
    ----------
    t0 : float in range [0, 1]
        Initial time parameter for the trajectories.
    t1 : float in range [0, 1]
        Final time parameter for the trajectories. Should satisfy t1 < t0
    a : ArrayLike of complex
        Array of initial points for the trajectories
    r : ArrayLike of complex
        Array of radii for the trajectories. Radius magnitude should be the length
        of the radius, and direction should be the direction to draw the circle
        from.
    turns : float or ArrayLike of float
        Number of turns to perform around the circle. If is an array, then it
        should have the same length as a, and provide the number of turns for
        each trajectory.
    phi0 : float in range [0, 2*pi]
        Initial phase on the circle.
    """
    def __init__(self, t0: float, t1: float, a: ArrayLike, r: ArrayLike,
                 turns: Union[float, ArrayLike], phi0:  Union[float, ArrayLike]):
        self.t0 = t0
        self.t1 = t1
        self.a = a
        self.r = r
        self.turns = turns
        self.phi0 = phi0

    def t_0(self, tau: float) -> ArrayLike:
        """
        Retrieves the time t(tau), following parameterized trajectories along a
        circle.

        Parameters
        ----------
        tau : float in range [t0, t1]
            Time trajectory parameter

        Returns
        -------
        t : ArrayLike of complex
            The time t(tau) on the trajectories.
        """
        dt = self.t1 - self.t0
        a = self.a - self.r * np.exp(self.phi0*1j)
        return a + np.exp(2j*np.pi*self.turns*(tau - self.t0)/dt + self.phi0*1j) * self.r

    def t_1(self, tau: float) -> ArrayLike:
        """
        Retrieves the derivative dt/dtau(tau), following parameterized trajectories
        along a circle.

        Parameters
        ----------
        tau : float in range [t0, t1]
            Time trajectory parameter

        Returns
        -------
        dt/dtau : ArrayLike of complex
            The derivative dt/dtau(tau) on the trajectories.
        """
        dt = self.t1 - self.t0
        return 2j*np.pi*self.turns/dt*self.r * \
            np.exp(2j*np.pi*self.turns*(tau - self.t0)/dt + self.phi0*1j)

    def length(self):
        return 2 * np.pi * np.abs(self.r * self.turns)

class LineTraj(TimeTrajectory):
    """
    Follows line trajectories in time

    .. math::
         t(\\tau) = a + (b - a) \\frac{\\tau - t_0}{t_1 - t_0}

    Parameters
    ----------
    t0 : float in range [0, 1]
        Initial time parameter for the trajectories.
    t1 : float in range [0, 1]
        Final time parameter for the trajectories. Should satisfy t1 < t0
    a : ArrayLike of complex
        Array of initial points for the trajectories
    b : ArrayLike of complex
        Array of final points for the trajectories
    """
    def __init__(self, t0: float, t1: float, a: ArrayLike, b: ArrayLike):
        self.t0 = t0
        self.t1 = t1
        self.a = a
        self.b = b

    def t_0(self, tau: float) -> ArrayLike:
        """
        Retrieves the derivative dt/dtau(tau), following parameterized trajectories
        along a line.

        Parameters
        ----------
        tau : float in range [t0, t1]
            Time trajectory parameter

        Returns
        -------
        t : ArrayLike of complex
            The time t(tau) on the trajectory.
        """
        dt = self.t1 - self.t0
        return self.a + (self.b - self.a)/dt * (tau - self.t0)

    def t_1(self, tau: float) -> ArrayLike:
        """
        Retrieves the derivative dt/dtau(tau), following parameterized trajectories
        along a line.

        Parameters
        ----------
        tau : float in range [t0, t1]
            Time trajectory parameter

        Returns
        -------
        dt/dtau : ArrayLike of complex
            The derivative dt/dtau(tau) on the trajectory.
        """
        dt = self.t1 - self.t0
        return (self.b - self.a)/dt

    def length(self):
        return np.abs(self.b - self.a)

######################################
#   Utility trajectory baseclasses   #
######################################
class SequentialTraj(TimeTrajectory):
    """
    Utility class for creating a trajectory from a sequence of trajectories.

    A subclass should provide two member parameters prior to the propagation:

    - path : A list holding the sequence of trajectories to connect. The assumption \
    is that the path runs from tau=0 to tau=1
    - discont_times : A list holding the parameteric time tau on which the \
        connection happens (and a discontinuity occours)

    Optionally, the class can provide the property 'my_discont_times', which will
    be used instead of discont_times. That is useful when having nested sequential
    trajectories, to differentiate between the total discontinuity points and
    the ones coming from this class.

    Parameters
    ----------
    t0 : float in range [0, 1]
        Initial time parameter for the trajectories.
    t1 : float in range [0, 1]
        Final time parameter for the trajectories. Should satisfy t1 < t0
    """

    discont_times: List[int]
    path: List[TimeTrajectory]

    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def t_0(self, tau):
        discont_times = self.discont_times
        if hasattr(self, 'my_discont_times'):
            discont_times = self.my_discont_times

        t = (tau - self.t0) / (self.t1 - self.t0)
        ts = np.array(discont_times + [1])
        path = np.flatnonzero(ts >= t)[0]
        return self.path[path].t_0(t)

    def t_1(self, tau):
        discont_times = self.discont_times
        if hasattr(self, 'my_discont_times'):
            discont_times = self.my_discont_times

        t = (tau - self.t0) / (self.t1 - self.t0)
        dt = 1 / (self.t1 - self.t0)                # Jacobian due to the change in tau
        ts = np.array(discont_times + [1])
        path = np.flatnonzero(ts >= t)[0]
        return self.path[path].t_1(t) * dt

    def get_discontinuity_times(self):
        return self.discont_times

    def length(self):
        return np.sum([p.length() for p in self.path], axis=0)

class OdeSolTraj(TimeTrajectory):
    """
    Utility trajectory class enveloping an ODE solution. Useful when the time
    trajectory is a result of a propagation of time on another coordinate like
    position or momentum.

    Parameters
    ----------
    t0 : float in range [0, 1]
        Initial time parameter for the trajectories.
    t1 : float in range [0, 1]
        Final time parameter for the trajectories. Should satisfy t1 < t0
    sol : Callable
        The propagation solution. When called as `sol(tau)` sould return the
        trajectory's coordinates at trajectory parameter `tau`
    n : int, optional
        Number of solutions to actually take from the beginning. Useful when
        the propagation result contains the trajectory in parameters other than
        time. If not positive then all solutions are taken. The default is 0.
    """
    def __init__(self, t0, t1, sol, n = 0):
        self.t0 = t0
        self.t1 = t1
        self.sol = sol
        self.n = n if n > 0 else sol(t0).shape[0]

    def t_0(self, tau):
        return self.sol(tau)[:self.n]

    def t_1(self, tau):
        dtau = np.finfo(np.float64).eps
        return ((self.sol(tau + dtau) - self.sol(tau)) / dtau)[:self.n]

    def length(self):
        taus = np.linspace(self.t0, self.t1, 3000)
        ts = np.stack([self.t_0(t) for t in taus])
        return np.reshape(np.trapz(np.abs(ts), taus, axis=0), (-1))
