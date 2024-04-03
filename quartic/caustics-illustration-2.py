# -*- coding: utf-8 -*-
"""
Illustration of the behavior of caustics and Stokes treatment we do to eliminate
unphysical regions, this time using an iterative method to locate the Stokes and
anti-Stokes lines as well.

The example propagates a Gaussian in quartic potential, the looks for the caustics
and applies Stokes treatment. It then plots the applied factor to each trajectory
(transparent to black color), the Stokes lines (green) and anti-Stokes lines (magenta)
for both caustics, and the diverging parts in blue.

In addition, this example finds the Stokes and anti-Stokes lines for each caustic
iteratively, and plots them for both caustics, Stokes lines in blue and anti-Stokes
in red.

Then it divides the sampled space into sectors according to the found lines,
and finds the sector to remove.

As this file contains many things, some might prove useful, one should consider
move things from here into the finco module.In addition, to control what is
plotted several switches are given at the first cell, and can be changed as needed.

@author: Noam Ottolenghi
"""
#%% Setup
import os
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh
from skimage.measure import points_in_poly

from quartic import S0, V, m, QuarticTimeTrajectory
from finco import create_ics, propagate, Mesh
from finco.stokes import separate_to_blobs, calc_factor2, approximate_F, find_caustics

plt.rc('font', size=14)

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.INFO)

logger = logging.getLogger('quartic.caustics_illustration')
logger.setLevel(logging.INFO)

# Switches
plot_lines1 = True # Plot lines for bottom-left caustic
plot_lines2 = True # Plot lines for upper-right caustic
plot_sectors1 = True # Plot sectors for bottom-left caustic. Overrides other displays
plot_sectors2 = False # Plot sectors for upper-right caustic. Overrides other displays

# Iteration parameters
dq = 1e-2

def find_phis(proj, caustic):
    """
    Locates the angles for both Stokes lines and anti-Stokes lines in nu-tilde space.

    Parameters
    ----------
    proj : DataFrame of projection map results
        Trajectories projection map to calculate angles from. Should be in the
        format outputted by FINCOresults.get_projection_map()
    caustic : pandas.Series
        A caustic to compute the angles for. Should be of the format outputted by
        find_caustics()

    Returns
    -------
    phis : list of 6 floats
        The 3 angles for the anti-Stokes lines and Stokes lines, concatenated.
    """
    _, F_3 = approximate_F(proj.q0, proj.xi, caustic)
    phi0 = np.angle(F_3)

    sphis = (np.arange(-4,4) * np.pi - phi0) / 3
    sphis = sphis[(sphis > -np.pi) & (sphis < np.pi)]
    aphis = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
    aphis = aphis[(aphis > -np.pi) & (aphis < np.pi)]
    return np.concatenate((aphis[:3], sphis[:3]))

def do_step(result, prev, stokes):
    """
    Performs the actual iteration, based on a very simple finite difference algorithm.

    Parameters
    ----------
    result : FINCOResults
        Propagation results from last step to calculate the next step for
    prev : ArrayLike of complex of shape (2, n_caustics, <lines>)
        Previous step's positions for a lines to iterate on.
    stokes : Arraylike of booleans of same shape as prev
        For each line, whether to iterate on the line as Stokes or anti-Stokes line.

    Returns
    -------
    step:
        Initial condition dataset for the next step, for propagation and reevaluation
        of the quantities needed for the iteration. The new positions are the
        propagation initial positions.

    Remarks
    -------

    The step is based on the following idea for iterating over Stokes and
    anti-Stokes lines. We are looking for curves in space where for two points
    :math:`q1,q2` on the curve satisfy

    .. math::
        \\Delta \\xi \\left( q_{1},q_{2} \\right) =
        \\xi \\left( q_{1} \\right) - \\xi \\left( q_{2} \\right) = 0

    and we have :math:`\\Im F(q_{1},q_{2}) = 0` for Stokes lines and
    :math:`\\Re F(q_{1},q_{2}) = 0` for anti-Stokes lines, where

    .. math::
        F\\left( q_{1}, q_{2} \\right) =
        \\sigma\\left( q_{1} \\right) - \\sigma\\left( q_{2} \\right) =
        \\sigma_{A}\\left( q_{1} \\right) - \\sigma_{A}\\left( q_{2} \\right)

    As such we can write the derivatives describing the change between these two points
    and F when enforcing :math:`\\Delta \\xi \\left( q_{1},q_{2} \\right) = 0`:

    .. math::
        \\left( \\frac{\\partial q_{1}}{\\partial q_{2}} \\right)_{\\Delta\\xi} &=
        \\frac{\\xi^{\\left( 1 \\right)} \\left( q_{2} \\right)}
        {\\xi^{\\left( 1 \\right)}\\left( q_{1} \\right)} \\\\
        \\left( \\frac{\\partial F}{\\partial q_{1}} \\right)_{\\Delta\\xi} &=
        \\sigma_{A}^{\\left( 1 \\right)}\\left( q_{1} \\right) -
        \\sigma_{A}^{\\left( 1 \\right)}\\left( q_{2} \\right)
        \\left( \\frac{\\partial q_{2}}{\\partial q_{1}} \\right)_{\\Delta\\xi} \\\\
        \\left( \\frac{\\partial F}{\\partial q_{2}} \\right)_{\\Delta\\xi} &=
        \\left( \\frac{\\partial q_{1}}{\\partial q_{2}} \\right)_{\\Delta\\xi}
        \\sigma_{A}^{\\left( 1 \\right)}\\left( q_{1} \\right) -
        \\sigma_{A}^{\\left( 1 \\right)}\\left( q_{2} \\right)

    Now by calculating those three quantities we can choose the directions to
    advance q1 and q2 to while keeping F purely real or imaginary, giving us
    two pairs of points (one for moving towards the caustic and one moving outward).
    By enforcing the condition for the first quantity we he two possible pairs,
    and we choose those the progress outwards from the caustic.
    """
    proj = result.get_projection_map(1)
    deriv = result.get_caustics_map(1)
    n_lines = int(len(proj) / 2) # Divide lines back into two to advance both sides.

    logger.debug('dxi: %f', (proj.xi.loc[1] - proj.xi.loc[0]).to_numpy())
    logger.debug('dsigma: %f', (proj.sigma.loc[1] - proj.sigma.loc[0]).to_numpy())
    logger.debug('dq0: %f', (proj.q0.loc[1] - proj.q0.loc[0]).to_numpy())

    # Calculate the second quantity in the remarks and choose dq1 in the direction
    # pointing outwards. Use the condition between dq1 and dq2 to find dq2
    sigma_1 = deriv.sigma_1.to_numpy()
    xi_1 = deriv.xi_1.to_numpy()
    F1 = sigma_1[:n_lines] - sigma_1[n_lines:] * xi_1[:n_lines] / xi_1[n_lines:]
    dq1 = (np.array([[1j, -1j], [1, -1]])[stokes.astype(int)].reshape(-1,2) *
           np.expand_dims((np.abs(F1) / F1 * dq), -1))
    dq1 = np.choose(np.argmax(np.abs(np.expand_dims(proj.q0.loc[:n_lines - 1], -1) +
                                     dq1 - np.expand_dims(prev[0].flatten(), -1)), axis=1), dq1.T)
    dq2 = xi_1[:n_lines] / xi_1[n_lines:] * dq1

    # Change step size if dq is big for one of q1, q2.
    norm = dq / np.max(np.abs([dq1, dq2]), axis=0)
    dq1 *= norm
    dq2 *= norm
    return create_ics(np.concatenate([deriv.q0.loc[:n_lines - 1] + dq1,
                                      deriv.q0.loc[n_lines:] + dq2]), S0 = S0)

def iterative_stokes(n_iters, phis, q, stokes):
    """
    Perform iterative finding of Stokes and anti-Stokes lines, based on initial
    guess of how the lines behave. The initial guess assumes the analysis we usually
    do to find the lines is very accurate close to the caustic, meaning that the
    angle in nu-tilde space should be enough.

    Parameters
    ----------
    n_iters : positive integer
        Number of iterations to preform.
    phis : Arraylike of floats of shape (n_caustics, <lines>)
        Initial angles in nu-tilde space for the lines. First axis should indicate
        to what caustic the angle is related.
    q : Arraylike of complex of shape (n_caustics, <lines>)
        Caustics to iterate for. All dimensions but the first should be 1
    stokes : Arraylike of booleans of same shape as phis
        For each line, whether to iterate on the line as Stokes or anti-Stokes line.

    Returns
    -------
    qs : Arraylike of complex shape (steps, 2, n_caustics, <lines>)
        For each line, the iteration results on the line. Divided into 2 parts,
        such that qs[i,0,...] refers to one part of the ith line (up to the caustic)
        and qs[i,1,...] refers to the other part (from the caustic). This division
        is given to ease locating sectors around the caustic.
    """
    phis, q, stokes = np.broadcast_arrays(phis, q, stokes)
    ics = create_ics(np.stack([q + dq*np.exp(1j*phis),
                      q - dq*np.exp(1j*phis)]).flatten(), S0 = S0)

    shape = np.concatenate([[2], q.shape])
    prev = np.stack([q, q])
    cur = propagate(ics,
                    V = V, m = m, gamma_f=1,
                    time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                    blocksize=1, n_jobs=6, verbose=False, trajs_path=None)
    qs = np.stack([prev, np.reshape(cur.q0.loc[:, 1], shape)])

    for i in tqdm(range(n_iters)):
        logger.debug('iteration %d', i+1)
        next_ics = do_step(cur, prev, stokes)
        prev = np.reshape(cur.q0[:,1], shape)
        cur = propagate(next_ics,
                        V = V, m = m, gamma_f=1,
                        time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                        blocksize=1, n_jobs=6, verbose=False, trajs_path=None)
        if i % 5 == 0:
            qs = np.concatenate([qs, np.reshape(cur.q0.loc[:, 1], np.concatenate([[1], shape]))])

    return qs

def find_boundary(points, corners):
    """
    Interpolates and returns the points on the boundary box completing for a given
    pair of points. one point should be inside the box and the other outside.

    Parameters
    ----------
    points : ArrayLike of 2 2D points (2x2 floats)
        The points to interpolate
    corners : ArrayLike of 4 2D points (4x2 floats)
        he corner of the boundary box. Should be a rectangular box parallel to the axes.

    Returns
    -------
    boundary : ArrayLike of 1 2D point (2x2 floats)
        The interpolated point

    """
    x_max, x_min = np.max(corners[:,0]), np.min(corners[:,0])
    y_max, y_min = np.max(corners[:,1]), np.min(corners[:,1])
    m = (points[1].imag - points[0].imag) / (points[1].real - points[0].real)
    if points[1].real > x_max:
        return np.array((x_max, m * (x_max - points[0].real) + points[0].imag))
    if points[1].real < x_min:
        return np.array((x_min, m * (x_min - points[0].real) + points[0].imag))
    if points[1].imag > y_max:
        return np.array((1 / m * (y_max - points[0].imag) + points[0].real, y_max))
    if points[1].imag < y_min:
        return np.array((1 / m * (y_min - points[0].imag) + points[0].real, y_min))
    return []

def find_pois(sline, A, S):
    """
    Interpolates from the sampled points where a Stokes line should pass,
    returning the points surrounding it. Used to locate the Stokes line on which
    the prefactor diverges in the sample.

    Parameters
    ----------
    sline : ArrayLike of 2D points
        Stokes line to interpolate for
    A : ArrayLike of integers in shape (n_triangles, 3, 3)
        The positions of the sampled points, divided into triangles and in barycentric
        coordinates.
    S : ArrayLike of integers in shape (n_triangles, 3)
        The indices of the sampled points, divided into triangles.

    Returns
    -------
    idxs : ArrayLike of integers
        The indices of the points on the triangles where the line is
    """
    B = np.stack((np.ones(sline.size), sline.real, sline.imag), axis=0)
    lams = np.linalg.pinv(A) @ B
    mask = np.all(lams >= 0,axis=1)
    return np.unique(np.concatenate([S[m] for m in mask.T]).flatten())

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass

#%% Propagate
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 251), np.linspace(-2.5, 2.5, 251))
q0s = (X+1j*Y).flatten()

result = propagate(create_ics(q0s, S0 = S0),
                   V = V, m = m, gamma_f=1,
                   time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                   blocksize=1024, n_jobs=5, verbose=True, trajs_path=None)


plt.figure('caustics-illustration-iterative', figsize=(8,6))

#%% Find caustics
deriv = result.get_caustics_map(1)
proj = result.get_projection_map(1)

blobs = separate_to_blobs(deriv, quantile=1e-2)
qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]

caustics = find_caustics(qs, V = V, m = m, S0 = S0,
                         time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3, n_jobs=3)

#%% Locate and isolate Stokes and anti-Stokes lines 1

F, F_3 = approximate_F(proj.q0, proj.xi, caustics.iloc[0])
phi0 = np.angle(F_3)

s_lines = []
sphis1 = (np.arange(-4,4) * np.pi - phi0)/3
sphis1 = sphis1[(sphis1 > -np.pi) & (sphis1 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - sphis1[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    s_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))

a_lines = []
aphis1 = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
aphis1 = aphis1[(aphis1 > -np.pi) & (aphis1 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - aphis1[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)

    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))

if plot_lines1:
    plt.plot(np.real(s_lines[0]), np.imag(s_lines[0]), ':g')
    plt.plot(np.real(s_lines[1]), np.imag(s_lines[1]), ':g')
    plt.plot(np.real(s_lines[2]), np.imag(s_lines[2]), ':g')
    plt.plot(np.real(a_lines[0]), np.imag(a_lines[0]), ':m')
    plt.plot(np.real(a_lines[1]), np.imag(a_lines[1]), ':m')
    plt.plot(np.real(a_lines[2]), np.imag(a_lines[2]), ':m')

#%% Locate and isolate Stokes and anti-Stokes lines 2

F, F_3 = approximate_F(proj.q0, proj.xi, caustics.iloc[1])
phi0 = np.angle(F_3)

s_lines = []
sphis2 = (np.arange(-4,4) * np.pi - phi0)/3
sphis2 = sphis2[(sphis2 > -np.pi) & (sphis2 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - sphis2[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    s_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))


a_lines = []
aphis2 = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
aphis2 = aphis2[(aphis2 > -np.pi) & (aphis2 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - aphis2[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)

    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))

if plot_lines2:
    plt.plot(np.real(s_lines[0]), np.imag(s_lines[0]), 'g')
    plt.plot(np.real(s_lines[1]), np.imag(s_lines[1]), 'g')
    plt.plot(np.real(s_lines[2]), np.imag(s_lines[2]), 'g')
    plt.plot(np.real(a_lines[0]), np.imag(a_lines[0]), 'm')
    plt.plot(np.real(a_lines[1]), np.imag(a_lines[1]), 'm')
    plt.plot(np.real(a_lines[2]), np.imag(a_lines[2]), 'm')

#%% Add Stokes treatment
S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
for (i, caustic) in caustics.iterrows():
    S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)

cmap = ListedColormap([[0,0,0,a] for a in np.linspace(1, 0)])
plt.scatter(np.real(proj.q0), np.imag(proj.q0), c=S_F, cmap=cmap, s=0.5)
plt.colorbar(label=r"$S(F)$")

#%% Iterative anti-stokes line

# We order the lines in (2,3) shape, one row for anti-Stokes lines and the other
# for Stokes lines. There are 3 lines of each kind, hence a (2,3) shape
q = np.reshape(caustics.q, (len(caustics),1,1))
phis = np.concatenate([find_phis(proj, caustic[1])
                       for caustic in caustics.iterrows()]).reshape((len(caustics),2,3))
stokes = np.reshape(([False]*3 + [True]*3) * len(caustics), (len(caustics),2,3))
qss = iterative_stokes(n_iters=550, phis=phis, q=q, stokes=stokes)

#%% Plot all found lines
astokes_it1 = [np.concatenate([qss[::-1,0,0,0,i], qss[:,1,0,0,i]]) for i in range(3)]
astokes_it2 = [np.concatenate([qss[::-1,0,1,0,i], qss[:,1,1,0,i]]) for i in range(3)]
stokes_it1 = [np.concatenate([qss[::-1,0,0,1,i], qss[:,1,0,1,i]]) for i in range(3)]
stokes_it2 = [np.concatenate([qss[::-1,0,1,1,i], qss[:,1,1,1,i]]) for i in range(3)]

if plot_lines1:
    plt.plot(np.real(astokes_it1[0]), np.imag(astokes_it1[0]), ':r')
    plt.plot(np.real(astokes_it1[1]), np.imag(astokes_it1[1]), ':r')
    plt.plot(np.real(astokes_it1[2]), np.imag(astokes_it1[2]), ':r')
    plt.plot(np.real(stokes_it1[0]), np.imag(stokes_it1[0]), ':b')
    plt.plot(np.real(stokes_it1[1]), np.imag(stokes_it1[1]), ':b')
    plt.plot(np.real(stokes_it1[2]), np.imag(stokes_it1[2]), ':b')

if plot_lines2:
    plt.plot(np.real(astokes_it2[0]), np.imag(astokes_it2[0]), 'r')
    plt.plot(np.real(astokes_it2[1]), np.imag(astokes_it2[1]), 'r')
    plt.plot(np.real(astokes_it2[2]), np.imag(astokes_it2[2]), 'r')
    plt.plot(np.real(stokes_it2[0]), np.imag(stokes_it2[0]), 'b')
    plt.plot(np.real(stokes_it2[1]), np.imag(stokes_it2[1]), 'b')
    plt.plot(np.real(stokes_it2[2]), np.imag(stokes_it2[2]), 'b')

#%% Identify sectors according to the iterative algorithm.

# This code showcases how the sectors can be identified using tools akin to image
# processing. Plotting is turned on by switch

# Determine points to check for
res = result.get_results(1)
pts = np.stack((np.real(res.q0), np.imag(res.q0)), axis=1)

# Set corners
corners = np.array([[2.5,2.5],[2.5,-2.5],[-2.5,-2.5],[-2.5,2.5]])

# Determine anti-Stokes lines
alines = qss[:,:,0,0,:].reshape(qss.shape[0],-1).T
masks = [np.max([np.abs(line.real), np.abs(line.imag)], axis=0) < 2.5 for line in alines]

# Determine sectors
sectors = np.zeros(pts.shape[0], dtype=int)
for i, (l1, l2, m1, m2) in enumerate(zip(alines[:-1], alines[1:], masks[:-1], masks[1:])):
    # Determine parts coinciding with boundaries
    p1 = np.array([l1[np.where(m1)[0][-1]], l1[np.where(~m1)[0][0]]])
    p2 = np.array([l2[np.where(m2)[0][-1]], l2[np.where(~m2)[0][0]]])
    bounds = np.array([find_boundary(p1, corners), find_boundary(p2, corners)])
    bounds = bounds[:,0] + 1j * bounds[:,1]

    # Add corners in the boundary
    in_sector = corners[(np.angle((corners[:,0] + 1j * corners[:,1]) / bounds[0]) > 0) &
                        (np.angle((corners[:,0] + 1j * corners[:,1]) / bounds[1]) < 0)]
    bounds = np.insert(bounds, 1, in_sector[:,0] + 1j * in_sector[:,1])

    # Determine polygon surrounding sector and points within it
    verts = np.concatenate((l1[m1], bounds, l2[m2][::-1]))
    poly = np.stack((verts.real, verts.imag), axis=1)

    sectors[points_in_poly(pts, poly)] = i + 1

#%% Find points around Stokes line and locate sectors to remove

# Calculate the triangles for intepolation. Done once for efficiency
mesh = Mesh(res)
qs = np.reshape(res.q0.take(mesh.tri.simplices.flatten()), mesh.tri.simplices.shape)
A = np.stack((np.ones(qs.shape), np.real(qs), np.imag(qs)), axis=1)

# Locate points closest to the Stokes lines
slines = qss[:,:,0,1,:].reshape(qss.shape[0],-1).T
pois = [mesh.mesh_to_points(find_pois(sline, A, mesh.tri.simplices)) for sline in slines]

# Locate stoke lines on which we have diverging values
idx = np.where([np.any(np.real(proj.loc[poi].sigma) > 0) for poi in pois])[0]

# Locate sectors to remove. Do that by identifying the sector of the farthest
# point from the caustic on the line, assuming sectors are better defined further
# from the caustic
toremove = [sectors[pois[i][np.argmax(np.abs(res.q0.loc[pois[i]].to_numpy() - caustics.iloc[0].q))]]
            for i in idx]
S_F = np.ones(len(res.q0))
S_F[np.isin(sectors, toremove)] = 0

if plot_sectors1:
    plt.tripcolor(q0s.real, q0s.imag, sectors)
    plt.scatter(np.real(proj.q0), np.imag(proj.q0), c=S_F, cmap=cmap, s=0.5)

#%% Identify sectors according to the iterative algorithm 2

# Determine lines
alines = qss[:,:,1,0,:].reshape(qss.shape[0],-1).T
masks = [np.max([np.abs(line.real), np.abs(line.imag)], axis=0) < 2.5 for line in alines]

# Determine sectors
sectors = np.zeros(pts.shape[0], dtype=int)
for i, (l1, l2, m1, m2) in enumerate(zip(alines[:-1], alines[1:], masks[:-1], masks[1:])):
    # Determine parts coinciding with boundaries
    p1 = np.array([l1[np.where(m1)[0][-1]], l1[np.where(~m1)[0][0]]])
    p2 = np.array([l2[np.where(m2)[0][-1]], l2[np.where(~m2)[0][0]]])
    bounds = np.array([find_boundary(p1, corners), find_boundary(p2, corners)])
    bounds = bounds[:,0] + 1j * bounds[:,1]

    # Add corners in the boundary
    in_sector = corners[(np.angle((corners[:,0] + 1j * corners[:,1]) / bounds[0]) > 0) &
                        (np.angle((corners[:,0] + 1j * corners[:,1]) / bounds[1]) < 0)]
    bounds = np.insert(bounds, 1, in_sector[:,0] + 1j * in_sector[:,1])

    # Determine polygon surrounding sector and points within it
    verts = np.concatenate((l1[m1], bounds, l2[m2][::-1]))
    poly = np.stack((verts.real, verts.imag), axis=1)

    sectors[points_in_poly(pts, poly)] = i + 1

#%% Find points around Stokes line and locate sectors to remove 2

# Locate points closest to the Stokes lines
slines = qss[:,:,1,1,:].reshape(qss.shape[0],-1).T
pois = [mesh.mesh_to_points(find_pois(sline, A, mesh.tri.simplices)) for sline in slines]

# Locate stoke lines on which we have diverging values
idx = np.where([np.any(np.real(proj.loc[poi].sigma) > 0) for poi in pois])[0]

# Locate sectors to remove. Do that by identifying the sector of the farthest
# point from the caustic on the line, assuming sectors are better defined further
# from the caustic
toremove = [sectors[pois[i][np.argmax(np.abs(res.q0.loc[pois[i]].to_numpy() - caustics.iloc[0].q))]]
            for i in idx]
S_F2 = np.ones(len(res.q0))
S_F2[np.isin(sectors, toremove)] = 0

if plot_sectors2:
    plt.tripcolor(q0s.real, q0s.imag, sectors)
    plt.scatter(np.real(proj.q0), np.imag(proj.q0), c=S_F2, cmap=cmap, s=0.5)

#%% Add diverging part
diverging_part = np.real(proj.sigma) > 0

c = [(0,0.7,0.9,int(d)) for d in diverging_part]
tri = Triangulation(np.real(proj.q0), np.imag(proj.q0))
col = TriMesh(tri, facecolors=c, edgecolors='face')
plt.gca().add_collection(col)
plt.gca().autoscale_view()

#%% Image finalization
plt.xlim(-2.5,2.5)
plt.xlabel(r'$\Re q_0$')
plt.ylim(-2.5,2.5)
plt.ylabel(r'$\Im q_0$')
plt.tight_layout()
plt.savefig('caustics-exploration/caustics-illustration-iterative.png')
