"""
Prototype of position trajectory manipulation for locating poles.

The script is a prototype of locating the poles in oscillating field by mutating
the radius of circumnavigation for points. The procedure is as follows:
    
    1. An initial guess based on a constant field is taken (here for 10 poles
       and only pole jumps)
    2. The trajectory in time is built, and a set points on it are chosen.
    3. For each point it is calculated where it is w.r.t. the two radii at the
    current time and position, and a small manipluation to the radius is added
    to "push" the trajectory the two radii.
    
Remarks
=======

    1. This is a prototype. It does not necessarily work and requires more effort
    for it to become usable. Should be worked upon.
    2. The code is not ordered yet as a script but a collection of cells (starting
    with #%%). Those can be run independently in most IDEs. Each cell has a title
    showing what it does. A typical run would probably be:
        "Setup" -> "Initial step" -> "Plot initial step"
    and then "Do step" repeatedly with running "Plot step" to plot the last step.

@author: Noam Ottolenghi
"""
#%% Setup

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import matplotlib.pyplot as plt

from coulomb import m, A0, q_e, V as V_field

from finco import create_ics
from finco.time_traj import SequentialTraj, LineTraj, CircleTraj, TimeTrajectory
from finco.coord2time import Space2TimeTraj
    
def S0_mock(_):
    """
    Mock S0 funcion to manually set initial momenta

    Parameters
    ----------
    q0 : ArrayLike
        Initial positions.
    """
    return p0

S0 = [S0_mock] * 3

class TestTraj(SequentialTraj):
    def __init__(self, r, n):
        super().__init__(t0=0, t1=1)
        self.r = r
        self.n = n

    def init(self, ics):
        q0 = ics.q.to_numpy()
        r = np.abs(self.r / q0) * q0
        self.path = [LineTraj(t0=0, t1=0.1, a=q0, b=r),
                     CircleTraj(t0=0.1, t1=1, a=r, r=r, turns=self.n, phi0=0)]
        self.discont_times = [0.1]

        return self

class TrajFromPoints(TimeTrajectory):
    def __init__(self, s, qs):
        self.s = s
        self.interp = interp1d(s, qs)
        
    def init(self, ics):
        return self
        
    def t_0(self, tau):
        return self.interp(tau)
    
    def t_1(self, tau):
        dtau = np.finfo(np.float64).eps
        if tau + dtau > 1:
            return -(self.interp(tau) - self.interp(tau - dtau)) / dtau
        return (self.interp(tau + dtau) - self.interp(tau)) / dtau
    
    def get_discontinuity_times(self):
        return []

def get_rs(A0, q0, p0):
    """
    Returns the two radii for given initial conditions and field strength.

    Parameters
    ----------
    A0 : complex
        Field strength.
    q0 : complex
        Initial position.
    p0 : complex
        Initial momentum.

    Returns
    -------
    rs : A 2-tuple of float
        The two calculated radii, with the first corresponding to
            .. math:: q_+ = \\frac{-E+\\sqrt{E^2-4eA_t}}{2A_t}
        and the second corresponding to
            .. math:: q_- = \\frac{-E-\\sqrt{E^2-4eA_t}}{2A_t}
    """
    E0 = p0**2/2/m - q_e/q0 - A0*q0
    qstars = np.stack([(-E0+(E0**2-4*A0*q_e)**0.5)/2/A0,
                       (-E0-(E0**2-4*A0*q_e)**0.5)/2/A0])
    return np.abs(qstars)


q0 = np.array([1-1j])
p0 = np.array([1+1j])
ics = create_ics(q0, S0)

n = 10
E0 = np.array(ics.p**2/2/m + V_field[0](ics.q,0))
qstars = np.concatenate([(-E0+(E0**2-4*A0*q_e)**0.5)/2/A0,
                         (-E0-(E0**2-4*A0*q_e)**0.5)/2/A0])
qstars = qstars[np.argsort(np.abs(qstars))]
rs = np.abs(qstars) * q0 / np.abs(q0)
r_nojump = rs[0] * 0.9
r_jump = 0.9 * rs[0] + 0.1 * rs[1]
r_field = rs[1] / 0.95

s = np.linspace(0,1,10000)

#%% Initial step
jump = Space2TimeTraj(t0=0, t1=1, q_traj=TestTraj(r_jump, n=10), V=V_field,
                      m=m, max_step=1e-4).init(ics)

#%% Plot initial step
ts = np.array([jump.t_0(_s) for _s in s])
qs = np.array([jump.q_traj.t_0(_s) for _s in s])
ps = np.array([jump.p(_s) for _s in s])

_, (q, p, tau) = plt.subplots(1, 3, num='const_jump', figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0, 0, c='r')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_ylabel(r'$\Im q$')

vals = p.scatter(ps.real, ps.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_ylabel(r'$\Im t$')

plt.tight_layout()

new_jump = jump
#%% Do step

s = np.abs(ps)
s = np.cumsum(np.abs(ps))
s -= s[0]
s /= s[-1]
ts = np.array([new_jump.t_0(_s) for _s in s])
qs = np.array([new_jump.q_traj.t_0(_s) for _s in s])
ps = np.array([new_jump.p(_s) for _s in s])

As = (V_field[0](qs, ts) + q_e / qs ) / qs
rs = np.sort(get_rs(As, qs, ps)[:,1000:,0], axis=0)
inds0 = np.where(np.count_nonzero(np.abs(qs[1000:,0]) > rs, axis=0) == 0)[0]
inds2 = np.where(np.count_nonzero(np.abs(qs[1000:,0]) > rs, axis=0) == 2)[0]

r_jumps0 = 0.5 * rs[0,inds0] + 0.5 * rs[1,inds0]
r_jumps2 = 0.5 * rs[0,inds2] + 0.5 * rs[1,inds2]

drs = np.zeros(len(qs))
drs[inds0 + 1000] = r_jumps0 - np.abs(qs[inds0 + 1000,0])
drs[inds2 + 1000] = r_jumps2 - np.abs(qs[inds2 + 1000,0])
drs = convolve(drs, [1/20]*20, 'same')
new_qs = qs[:,0] + drs * (qs / np.abs(qs))[:,0] * 1
max_diff = np.max(np.abs(drs))

new_q_traj = TrajFromPoints(s, np.expand_dims(new_qs, axis=0))
new_jump = Space2TimeTraj(t0=0, t1=1, q_traj=new_q_traj, V=V_field,
                      m=m, max_step=5e-5).init(ics)
s = np.linspace(0,1,10000)
ts = np.array([new_jump.t_0(_s) for _s in s])
qs = np.array([new_jump.q_traj.t_0(_s) for _s in s])
ps = np.array([new_jump.p(_s) for _s in s])

#%% Plot step

_, (q, p, tau) = plt.subplots(1, 3, figsize=(14,4))
vals = q.scatter(qs.real, qs.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=q)
q.scatter(0, 0, c='r')
q.set_title(r'$q$')
q.set_xlabel(r'$\Re q$')
q.set_ylabel(r'$\Im q$')

vals = p.scatter(ps.real, ps.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=p)
p.set_title(r'$p$')
p.set_xlabel(r'$\Re p$')
p.set_ylabel(r'$\Im p$')

vals = tau.scatter(ts.real, ts.imag, c=s, s=3, cmap='cool')
plt.colorbar(vals, label='trajectory parameter', ax=tau)
tau.set_title(r'$t$')
tau.set_xlabel(r'$\Re t$')
tau.set_ylabel(r'$\Im t$')

plt.tight_layout()
