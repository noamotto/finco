# -*- coding: utf-8 -*-
"""
Simple mesh data structure for adaptive sampling of phase space.

The mesh in its core is a data structure comprised of a dictionary between each
point and its neighbors on the mesh. It does not hold information abot the 
trajectories themselves.
    
The mesh supports several convenience methods for connecting points together and
disconnecting them, as well as adding more points to the mesh.

A typical workflow will be therefore:
    1. Sample a set of initial states (using create_ics() for example)
    2. Add these points to the mesh using add_points(). This will assign indices
        to the points based on those alreay in the mesh
    3. Connect neighboring points together using connect()
    4. Propagate the trajectories corresponding to each point
"""

import numpy as np
import pandas as pd
import joblib
from scipy.spatial import Delaunay
from functools import reduce

class Mesh:
    """
    Simple mesh data structure for adaptive sampling of phase space.
    
    Creates a mesh from a given set of points, as described in create_ics(). 
    The mesh is created by triangulsation using Delaunay's algorithm. The 
    object also allows simple conversion between indices of points as given in 
    the dataset and indices given by the mesh.
    
    Parameters
    ----------
    ics : pandas.DataFrame
        The points to create the mesh from. Should have the same index as returned
        from create_ics(), and only one timestep.
    
    adaptive : bool, optional
        Whether the given mesh is used for adaptive sampling or not, and should 
        allow adding more points to the mesh. The default is False.
    """
    def __init__(self, ics, adaptive=False):
        self.adaptive = adaptive
        self.ptm = {ics.index[i][0] : i for i in range(len(ics))}
        self.mtp = {i : ics.index[i][0] for i in range(len(ics))}
        
        points = np.stack([np.real(ics.q0), np.imag(ics.q0)], axis=-1)
        if not adaptive:
            qhull_options = "Qbb Qc Qz Q12"
        else:
            qhull_options = "Qc"
            
        self.tri = Delaunay(points, incremental=adaptive, qhull_options=qhull_options)
    
    @property
    def triangles(self):
        """
        Array of the triangle simplices in the mesh
        """
        return self.tri.simplices
    
    def get_neighbors(self, point_index, connectivity=1):
        mesh_ind = self.ptm[point_index]
        indptr, indices = self.tri.vertex_neighbor_vertices
        mesh_neighbors = indices[indptr[mesh_ind]:indptr[mesh_ind+1]]
        
        if connectivity == 1:
            return {self.mtp[i] for i in mesh_neighbors}
        if len(mesh_neighbors) == 0:
            return set()
        
        return set.union(*[self.get_neighbors(i, connectivity - 1) 
                           for i in mesh_neighbors]) - {point_index}
    
    def get_neighbors_value(self, value, points=None):
        """
        Returns the values corresponfing to the neighbors of each point on the
        mesh.

        Parameters
        ----------
        value : pandas.Series
            Series of values to take the values from. Should have the same order
            as the dataset used to create the mesh.
        points : ArrayLike, optional
            List of points to calculate the neighbor values for. None means 
            return all points.

        Returns
        -------
        values: list
            List of series, with the neighbor values of each point.
        """
        def stretch(x, y):
            rng, ind = y
            x[rng] = ind
            return x
        
        indptr, indices = self.tri.vertex_neighbor_vertices
        values = pd.DataFrame(value.take(indices))
        mpoints = (self.points_to_mesh(points) if points is not None 
            else range(len(self.tri.points)))
        if points is None:
            points = self.mesh_to_points(mpoints)
        ranges = [np.arange(indptr[i], indptr[i + 1]) for i in mpoints]
        values['point'] = reduce(stretch, zip(ranges, points), np.array([np.nan]*len(indices)))
        values = values.dropna().set_index('point', append=True)
        return values.reorder_levels(['point', 't_index', 'timestep'])
    
    def __getitem__(self, point_index):
        """
        Return the neighbors of a point, given point indices

        Parameters
        ----------
        point_index : integer
            The point's index, in the dataset indices.

        Returns
        -------
        neighbors: set
            The point's neighbors, in the dataset indices.
        """
        return self.get_neighbors(point_index)
    
    def points_to_mesh(self, points):
        points = np.array(points)
        return np.reshape([self.ptm[p] for p in np.array(points).flatten()], points.shape)
    
    def mesh_to_points(self, mpoints):
        mpoints = np.array(mpoints)
        return np.reshape([self.mtp[p] for p in np.array(mpoints).flatten()], mpoints.shape)
    
    def add_points(self, new_points: pd.DataFrame):
        """
        Adds a set of points into the mesh. 
        
        This is done by adding the new points as entries to the neighbors dictionary,
        with new indices given by the mesh to the points, and referring to the
        indices of these points on the mesh.
        
        As a result, these points will not be connected, and should be connected
        to neighbors later.

        Parameters
        ----------
        new_points : pd.DataFrame
            New points to add to the mesh. Should have a similar index format to the 
            datasets created by create_ics(). 

        Returns
        -------
        indices_map: dict
            A dictionary mapping between the indices in new_points to the indices
            of the points on the mesh. Can be used to connect the added points to
            those on the mesh.
        mesh_points: pandas DataFrame
            The added points with the indices given to them by the mesh. Basically 
            this is the same as new_points, but with updated index.
        """
        start_ind = np.max(list(self.ptm.keys())) + 1 if len(self.ptm) > 0 else 0

        new_ics = new_points.loc[(slice(None), 0),:]
        # mesh_index = pd.MultiIndex.from_tuples([(a[0] + start_ind, a[1]) for a in new_index],
        #                                        names=['t_index', 'timestep'])
        mesh_index = np.arange(len(new_ics)) + start_ind
        
        new_map = {new_ics.index[i][0] : mesh_index[i] for i in range(len(mesh_index))}
        new_index = pd.MultiIndex.from_tuples([(new_map[ind[0]], ind[1]) for ind in new_points.index],
                                                names=['t_index', 'timestep'])
        identity = {mesh_index[i] : mesh_index[i] for i in range(len(mesh_index))}
        
        self.ptm.update(identity)
        self.mtp.update(identity)
        
        points = np.stack([np.real(new_ics.q0), np.imag(new_ics.q0)], axis=-1)
        self.tri.add_points(points)
            
        return new_map, new_points.set_index(new_index)
        
    def save(self, filepath):
        """
        Saves the mesh into a file.

        Parameters
        ----------
        filepath : string
            Path to save the mesh at.
        """
        joblib.dump(self.neighbors, filepath, compress=True)
    
    
def load_mesh(filepath) -> Mesh:
    """
    Loads a mesh from a mesh file, saved using Mesh.save() .


    Parameters
    ----------
    filepath : string
        Path to load the mesh from.

    Returns
    -------
    mesh : Mesh
        The loaded mesh.
    """
    mesh = Mesh()
    mesh.neighbors = joblib.load(filepath)
    
    return mesh
