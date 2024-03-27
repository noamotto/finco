# -*- coding: utf-8 -*-
"""
Utility trajectories translating a trajectory in some coordinate in phase space
into a trajectory in time.

@author: Noam Ottolenghi
"""

from .time_traj import TimeTrajectory, SequentialTraj, OdeSolTraj

import numpy as np
from scipy.integrate import solve_ivp

class Space2TimeTraj(SequentialTraj):
    """
    Utility trajectory class creating a trajectory in time based on a trajectory in space,
    using the following two equations
    
    .. math::
         \\frac{dt}{dq} &= \\frac{m}{p} \\\\
         \\frac{dp}{dq} &= -\\nabla V(q,t) \\frac{m}{p}
    
    Parameters
    ----------
    t0 : float in range [0, 1]
        Initial time parameter for the trajectories.
    t1 : float in range [0, 1]
        Final time parameter for the trajectories. Should satisfy t1 < t0
    q_traj : TimeTrajectory
        The trajectory in position space, as :math:`q(tau)`
    V : tuple of 3 Callables
        The system's potential and its first two derivatives
    m : float
        The system's mass
    All other parameters are passed to the integrator. See `scipy.integrate.solve_ivp`
    for list of parameters.
    """
    def __init__(self, t0: float, t1: float, q_traj: TimeTrajectory,
                 V: list, m: float, **kwargs):
        super().__init__(t0, t1)
        
        self.q_traj = q_traj
        self.V = V
        self.m = m
        self.kwargs = kwargs
        
    def init(self, ics):
        def do_step(tau: float, y, q_trajs: list, V: list, m: float):
            q = q_trajs[0](tau)
            dq = q_trajs[1](tau)
            
            t, p = y.reshape(2, -1)
            dtdq = m / p

            return np.concatenate([dtdq, -V[1](q, t) * dtdq] * dq)

        self.res = []
        self.path = []
        
        self.q_traj = self.q_traj.init(ics)
        self.discont_times = self.q_traj.get_discontinuity_times()
        q0, p0, t0 = ics.q.to_numpy(), ics.p.to_numpy(), ics.t.to_numpy()
        
        y0 = np.concatenate([t0, p0])
        for t0, t1 in zip([0] + self.discont_times, self.discont_times + [1]):
            res = solve_ivp(do_step, t_span = (t0, t1),
                            y0 = y0, args=(self.q_traj.t_funcs, self.V, self.m), dense_output=True,
                            **self.kwargs)
            self.path.append(OdeSolTraj(t0, t1, res.sol, q0.size))
            
            y0 = res.y[:,-1]
            # self.res.append(res)
            
        return self
    