# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh

from finco import TimeTrajectory, create_ics, propagate
from finco.stokes import separate_to_blobs, find_caustics, calc_factor2, approximate_F
# from utils import tripcolor_complex

plt.rc('font', size=14)

logging.basicConfig()
logging.getLogger('finco').setLevel(logging.INFO)


# System params
m = 1
chi = 2j
gamma0 = 0.5
a = 0.5
b = 0.1

def S0_0(q):
    return -1j*(-gamma0 * (q-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0 + 0.25*np.log(2*gamma0/np.pi))
    
def S0_1(q):
    return -1j*(-2*gamma0 * (q-np.conj(chi)/2/gamma0))

def S0_2(q):
    return np.full_like(q, 2j*gamma0)

def V_0(q):
    return a*q**2 + b*q**4
    
def V_1(q):
    return 2*a*q + 4*b*q**3

def V_2(q):
    return 2*a + 12*b*q**2

class QuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics):        
        self.t = np.full_like(ics.q, 0.72)
        
    def t_0(self, tau):
        return self.t * tau
        
    def t_1(self, tau):
        return self.t

class HalfQuarticTimeTrajectory(TimeTrajectory):
    def init(self, ics): 
        self.t0 = ics.t.to_numpy()
        self.t = np.full_like(ics.q, 0.72/2) + self.t0
        
    def t_0(self, tau):
        return (self.t - self.t0) * tau + self.t0
        
    def t_1(self, tau):
        return self.t - self.t0

def eliminate_stokes(result):
    # Load projection map, map to a grid, and calculate F
    deriv = result.get_caustics_map(1)
    proj = result.get_projection_map(1)

    # plt.figure()
    # tripcolor_complex(np.real(proj.q0), np.imag(proj.q0), deriv.xi_1.to_numpy(), absmax=1e2)

    blobs = separate_to_blobs(deriv, quantile=1e-2)
    qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]
        
    caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2], 
                             time_traj=QuarticTimeTrajectory(), gamma_f=1, dt=1e-3)
    # caustics = caustics[np.real(caustics.q) > 0]
    
    S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
    for (i, caustic) in caustics.iterrows():
        # idx = np.argmin(np.abs(proj.q0-caustic.q))
        # caustic.q = proj.q[idx]
        # caustic.xi = proj.xi.iat[idx]
        S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)
    
    return S_F

try:
    os.mkdir('caustics-exploration')
except FileExistsError:
    pass 

#%% Propagate
X, Y = np.meshgrid(np.linspace(-3.5, 3.5, 501), np.linspace(-3.5, 3.5, 501))
qs = (X+1j*Y).flatten()

result = propagate(create_ics(qs, S0 = [S0_0, S0_1, S0_2], gamma_f=1), 
                   V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                   time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                   blocksize=1024, n_jobs=5, verbose=True, trajs_path=None)


# trajs = result.get_trajectories(1)
plt.figure('caustics-illustration')
# tripcolor_complex(np.real(trajs.q0), np.imag(trajs.q0), trajs.pref, absmax=1e7)

#%% Find caustic at bottom-right
deriv = result.get_caustics_map(1)
proj = result.get_projection_map(1)

blobs = separate_to_blobs(deriv, quantile=1e-2)
qs = [deriv.q0[deriv.xi_1.abs()[blob].idxmin()] for blob in blobs]

caustics = find_caustics(qs, V = [V_0, V_1, V_2], m = m, S0 = [S0_0, S0_1, S0_2], 
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

plt.plot(np.real(s_lines[0]), np.imag(s_lines[0]), ':g')
plt.plot(np.real(s_lines[1]), np.imag(s_lines[1]), ':g')
plt.plot(np.real(s_lines[2]), np.imag(s_lines[2]), ':g')

a_lines = []
aphis1 = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
aphis1 = aphis1[(aphis1 > -np.pi) & (aphis1 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - aphis1[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    # Remove incorrect line in top-right corner that messes up with illustration.
    # mask = ((dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2) & 
    #         ((np.real(proj.q0) < 1) | (np.imag(proj.q0) < 1)))
    
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))

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

plt.plot(np.real(s_lines[0]), np.imag(s_lines[0]), 'g')
plt.plot(np.real(s_lines[1]), np.imag(s_lines[1]), 'g')
plt.plot(np.real(s_lines[2]), np.imag(s_lines[2]), 'g')

a_lines = []
aphis2 = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
aphis2 = aphis2[(aphis2 > -np.pi) & (aphis2 < np.pi)]

dists = np.angle(F.v_t)[:,np.newaxis] - aphis2[np.newaxis,:]
dists = np.min(np.abs(np.stack([dists - 2*np.pi, dists, dists + 2*np.pi])), axis=0)

for i in range(3):
    mask = (dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2)
    # Remove incorrect line in top-right corner that messes up with illustration.
    # mask = ((dists[:, i] < 1e-2) | (dists[:, i+3] < 1e-2) & 
    #         ((np.real(proj.q0) < 1) | (np.imag(proj.q0) < 1)))
    
    v_t_dists = pd.Series((np.real(F.v_t[mask])*20).astype(int)/20, index=F[mask].index)
    a_lines.append(proj.q0[mask].groupby(by = lambda x: v_t_dists.loc[x]).apply(np.median))

plt.plot(np.real(a_lines[0]), np.imag(a_lines[0]), 'm')
plt.plot(np.real(a_lines[1]), np.imag(a_lines[1]), 'm')
plt.plot(np.real(a_lines[2]), np.imag(a_lines[2]), 'm')

#%% Add Stokes treatment
from matplotlib.colors import ListedColormap

S_F = pd.Series(np.ones_like(proj.xi, dtype=np.float64), index=proj.q0.index)
for (i, caustic) in caustics.iterrows():
    S_F *= calc_factor2(caustic, proj.q0, proj.xi, proj.sigma)

cmap = ListedColormap([[a,a,a] for a in np.linspace(0, 1)])
plt.scatter(np.real(proj.q0), np.imag(proj.q0), c=S_F, cmap=cmap, s=1)
plt.colorbar(label=r"$S(F)$")

#%% Add diverging part
diverging_part = np.real(proj.sigma) > 0

c = [(0,0.7,0.9,int(d)) for d in diverging_part]
tri = Triangulation(np.real(proj.q0), np.imag(proj.q0))
col = TriMesh(tri, facecolors=c, edgecolors='face')
plt.gca().add_collection(col)
plt.gca().autoscale_view()

#%% Iterative anti-stokes line 
from joblib import Parallel, delayed
from tqdm import tqdm
dq = 1e-2

def find_phis(proj, caustic):
    F, F_3 = approximate_F(proj.q0, proj.xi, caustic)
    phi0 = np.angle(F_3)

    sphis = (np.arange(-4,4) * np.pi - phi0)/3
    sphis = sphis2[(sphis2 > -np.pi) & (sphis2 < np.pi)]
    aphis = (np.arange(-4,4) * np.pi - phi0)/3 + np.pi/6
    aphis = aphis2[(aphis2 > -np.pi) & (aphis2 < np.pi)]
    return np.concatenate((aphis[:3], sphis[:3]))

def do_step(result, prev, stokes):
    proj = result.get_projection_map(1)
    deriv = result.get_caustics_map(1)
    n = int(len(proj) / 2)
    # print(f'dxi: {(proj.xi.loc[1] - proj.xi.loc[0]).to_numpy()}')
    # print(f'dsigma: {(proj.sigma.loc[1] - proj.sigma.loc[0]).to_numpy()}')
    # print(f'dq0: {(proj.q0.loc[1] - proj.q0.loc[0]).to_numpy()}')
    sigma_1 = deriv.sigma_1.to_numpy()
    xi_1 = deriv.xi_1.to_numpy()
    F1 = sigma_1[:n] - sigma_1[n:] * xi_1[:n] / xi_1[n:]
    dq1 = (np.array([[1j, -1j], [1, -1]])[stokes.astype(int)].reshape(-1,2) * 
           np.expand_dims((np.abs(F1) / F1 * dq), -1))
    dq1 = np.choose(np.argmax(np.abs(np.expand_dims(proj.q0.loc[:n-1], -1) + 
                                     dq1 - np.expand_dims(prev[0].flatten(), -1)), axis=1), dq1.T)
    dq2 = xi_1[:n] / xi_1[n:] * dq1
    norm = dq / np.max(np.abs([dq1, dq2]), axis=0)
    dq1 *= norm
    dq2 *= norm
    return create_ics(np.concatenate([deriv.q0.loc[:n-1] + dq1,
                                      deriv.q0.loc[n:] + dq2]), S0 = [S0_0, S0_1, S0_2], gamma_f=1)    

def iterative_stokes(n, phis, q, stokes):
    phis, q, stokes = np.broadcast_arrays(phis, q, stokes)
    ics = create_ics(np.stack([q + dq*np.exp(1j*phis), 
                      q - dq*np.exp(1j*phis)]).flatten(), S0 = [S0_0, S0_1, S0_2], gamma_f=1)
    
    shape = np.concatenate([[2], q.shape])
    prev = np.stack([q, q])
    cur = propagate(ics, 
                    V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                    time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                    blocksize=1, n_jobs=6, verbose=False, trajs_path=None)
    qs = np.stack([prev, np.reshape(cur.q0.loc[:, 1], shape)])
    
    for i in tqdm(range(n)):
        # print(f'iteration {i+1}')
        next_ics = do_step(cur, prev, stokes)
        prev = np.reshape(cur.q0[:,1], shape)
        cur = propagate(next_ics, 
                        V = [V_0, V_1, V_2], m = m, gamma_f=1, 
                        time_traj = QuarticTimeTrajectory(), dt = 1e-3, drecord=1,
                        blocksize=1, n_jobs=6, verbose=False, trajs_path=None)
        if i % 5 == 0:
            qs = np.concatenate([qs, np.reshape(cur.q0.loc[:, 1], np.concatenate([[1], shape]))])
    
    return qs

qs = np.reshape(caustics.q, (len(caustics),1,1))
phis = np.concatenate([find_phis(proj, caustic[1]) for caustic in caustics.iterrows()]).reshape(len(caustics),2,3)
stokes = np.reshape(([False]*3 + [True]*3) * len(caustics), (len(caustics),2,3))
qss = iterative_stokes(n=550, phis=phis, q=qs, stokes=stokes)

# qss1 = iterative_stokes(n=550, phis=np.concatenate((aphis1[:3], sphis1[:3])), q=caustics.loc[0].q,
#                         stokes=np.array([False]*3 + [True]*3))
# qss2 = iterative_stokes(n=550, phis=np.concatenate((aphis2[:3], sphis2[:3])), q=caustics.loc[1].q,
#                         stokes=np.array([False]*3 + [True]*3))
#%%
astokes_it1 = [np.concatenate([qss[::-1,0,0,0,i], qss[:,1,0,0,i]]) for i in range(3)]
astokes_it2 = [np.concatenate([qss[::-1,0,1,0,i], qss[:,1,1,0,i]]) for i in range(3)]

plt.plot(np.real(astokes_it1[0]), np.imag(astokes_it1[0]), ':r')
plt.plot(np.real(astokes_it1[1]), np.imag(astokes_it1[1]), ':r')
plt.plot(np.real(astokes_it1[2]), np.imag(astokes_it1[2]), ':r')
plt.plot(np.real(astokes_it2[0]), np.imag(astokes_it2[0]), 'r')
plt.plot(np.real(astokes_it2[1]), np.imag(astokes_it2[1]), 'r')
plt.plot(np.real(astokes_it2[2]), np.imag(astokes_it2[2]), 'r')

#%% Identify sectors according to the iterative algorithm
from skimage.measure import points_in_poly

def find_boundary(points, corners):
    x_max, x_min = np.max(corners[:,0]), np.min(corners[:,0])
    y_max, y_min = np.max(corners[:,1]), np.min(corners[:,1])
    m = (points[1].imag - points[0].imag) / (points[1].real - points[0].real)
    if points[1].real > x_max:
        return np.array((x_max, m * (x_max - points[0].real) + points[0].imag))
    elif points[1].real < x_min:
        return np.array((x_min, m * (x_min - points[0].real) + points[0].imag))
    elif points[1].imag > y_max:
        return np.array((1 / m * (y_max - points[0].imag) + points[0].real, y_max))
    elif points[1].imag < y_min:
        return np.array((1 / m * (y_min - points[0].imag) + points[0].real, y_min))
    return []

# Determine points to check for
res = result.get_results(1)
pts = np.stack((np.real(res.q0), np.imag(res.q0)), axis=1)

# Set corners
corners = np.array([[2.5,2.5],[2.5,-2.5],[-2.5,-2.5],[-2.5,2.5]])

# Determine lines
lines = qss1[:,:,:3].reshape(qss1.shape[0],-1).T
masks = [np.max([np.abs(line.real), np.abs(line.imag)], axis=0) < 2.5 for line in lines]

# Determine sectors
sectors = np.zeros(pts.shape[0], dtype=int)
for i, (l1, l2, m1, m2) in enumerate(zip(lines[:-1], lines[1:], masks[:-1], masks[1:])):
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
from finco import Mesh

def find_pois(sline, mesh, A, S):
    B = np.stack((np.ones(sline.size), sline.real, sline.imag), axis=0)
    lams = np.linalg.pinv(A) @ B
    mask = np.all(lams >= 0,axis=1)
    return mesh.mesh_to_points(np.unique(np.concatenate([S[m] for m in mask.T]).flatten()))

# Locate points closest to the Stokes lines
mesh = Mesh(res)
qs = np.reshape(res.q0.take(mesh.tri.simplices.flatten()), mesh.tri.simplices.shape)
A = np.stack((np.ones(qs.shape), np.real(qs), np.imag(qs)), axis=1)
slines = qss1[:,:,3:].reshape(qss1.shape[0],-1).T
pois = [find_pois(sline, mesh, A, mesh.tri.simplices) for sline in slines]

# Locate stoke lines on which we have diverging values
idx = np.where([np.any(np.real(proj.loc[poi].sigma) > 0) for poi in pois])[0]

# Locate sectors to remove. Do that by identifying the sector of the farthest
# point from the caustic on the line, assuming sectors are better defined further
# from the caustic
toremove = [sectors[pois[i][np.argmax(np.abs(res.q0.loc[pois[i]].to_numpy() - caustics.iloc[0].q))]]
            for i in idx]
S_F = np.ones(len(res.q0))
S_F[np.isin(sectors, toremove)] = 0


#%% Identify sectors according to the iterative algorithm 2

# Determine lines
lines = qss2[:,:,:3].reshape(qss2.shape[0],-1).T
masks = [np.max([np.abs(line.real), np.abs(line.imag)], axis=0) < 2.5 for line in lines]

# Determine sectors
sectors = np.zeros(pts.shape[0], dtype=int)
for i, (l1, l2, m1, m2) in enumerate(zip(lines[:-1], lines[1:], masks[:-1], masks[1:])):
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
slines = qss2[:,:,3:].reshape(qss2.shape[0],-1).T
pois = [find_pois(sline, mesh, A, mesh.tri.simplices) for sline in slines]

# Locate stoke lines on which we have diverging values
idx = np.where([np.any(np.real(proj.loc[poi].sigma) > 0) for poi in pois])[0]

# Locate sectors to remove. Do that by identifying the sector of the farthest
# point from the caustic on the line, assuming sectors are better defined further
# from the caustic
toremove = [sectors[pois[i][np.argmax(np.abs(res.q0.loc[pois[i]].to_numpy() - caustics.iloc[0].q))]]
            for i in idx]
S_F2 = np.ones(len(res.q0))
S_F2[np.isin(sectors, toremove)] = 0

#%% Image finalization
plt.xlim(-2.5,2.5)
plt.xlabel(r'$\Re q_0$')
plt.ylim(-2.5,2.5)
plt.ylabel(r'$\Im q_0$')
plt.tight_layout()
plt.savefig('caustics-exploration/caustics-illustration-with-stokes.png')