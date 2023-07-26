# -*- coding: utf-8 -*-
"""
FINCO results handlers

The FINCO algorithm produces several result datasets, that can then be used to
analyse the propagation results and reconstruct the wavefunction. The datasets
are retrieved as pandas DataFrames, with index as specified in create_ics().

As writing new results is done solely by the propagation methods, usage of
FINCOWriter directly in scripts is uncommon. In order to load results please
use load_results(), or results_from_data() to wrap raw results.

The loading from a results file is done lazily, meaning that data is processed
on demand. For efficiency, one can load a portion of a results file into memory
using get_view().
"""

import os
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
from joblib import delayed, Parallel

from .utils import hbar
from .mesh import Mesh

def gf(x: ArrayLike, qf: float, pf: float, gamma_f: float) -> ArrayLike:
    """
    Utility function. Reconstructs a Gaussian from given parameters.

    Parameters
    ----------
    x : 1D ArrayLike of floats
        x positions to reconstruct the Gaussian for.
    qf : float
        Gaussian's center.
    pf : float
        Gaussian's momentum.
    gamma_f : float
        Gaussian width.

    Returns
    -------
    psi : 1D Array like in the shape of x
        The reconstructed Gaussian.

    """
    X, Qf = np.meshgrid(np.array(x), np.array(qf))
    Pf = np.array(pf).reshape(-1, 1)
    return (2*gamma_f / np.pi) ** 0.25 * np.exp(-gamma_f*(X-Qf)**2 + 1j / hbar * Pf * (X-Qf))

def _calc_projection(results: pd.DataFrame, gamma_f: float) -> [pd.Series, pd.Series,
                                                                pd.Series, pd.Series]:
    """
    Calculates Gaussian projection on the real axis parameters.

    Parameters
    ----------
    results : pandas.DataFrame
        Raw data to calculate from, as saved by FINCOWriter
    gamma_f : float
        Value of gamma_f used in propagation.

    Returns
    -------
    qf : pandas.Series of float
        Peak location of the trajectory's Gaussian projection on
        the real axis at timestep
    pf : pandas.Series of float
        Peak momentum of the trajectory's Gaussian projection on
        the real axis at timestep
    xi : pandas.Series of complex
        Trajectory's Hubber-Heller projection map at timestep
    sigma : pandas.Series of complex
        Exponent resulting from the projection as defined in the paper
    """
    xi = 2 * gamma_f * results.q - 1j / hbar * results.p
    qf, pf = np.real(xi) / 2 / gamma_f, -np.imag(xi) * hbar
    sigma = (1j / hbar * results.S +
                 (results.p**2 - pf**2) / 4 / gamma_f / hbar ** 2)

    return pd.Series(qf, index=results.index), pd.Series(pf, index=results.index), xi, sigma

def _calc_derivatives(results: pd.DataFrame, gamma_f: float) -> [pd.Series, pd.Series]:
    """
    Calculates derivatives of the Gaussian projection on the real axis parameters.

    Parameters
    ----------
    results : pandas.DataFrame
        Raw data to calculate from, as created by create_ics()
    gamma_f : float
        Value of gamma_f used in propagation.

    Returns
    -------
    xi_1 : pandas.Series of complex
        First defivative of xi w.r.t. q0
    sigma_1 : pandas.Series of complex
        First defivative of sigma w.r.t. q0
    """
    xi_1 = results.xi_1_abs * np.exp(results.xi_1_angle * 1j)
    sigma_1 = 1j / 2 / gamma_f / hbar * results.p * xi_1
    return xi_1, sigma_1

def _calc_pref(results: pd.DataFrame, gamma_f: float) -> pd.Series:
    """
    Calculates the prefactor of trajectories, their contribution to the integral,
    at timestep

    Parameters
    ----------
    results : pandas.DataFrame
        Raw data to calculate from, as saved by FINCOWriter
    gamma_f : float
        Value of gamma_f used in propagation.

    Returns
    -------
    pref : ArrayLike of complex
        The prefactor of trajectories, their contribution to the integral,
        at timestep
    """
    *_, sigma = _calc_projection(results, gamma_f)

    pref = (results.xi_1_abs ** 1.5 *             # Jacobian's norm
            np.exp(results.xi_1_angle * -0.5j) *  # Jacobian's angle
            np.exp(sigma))                        # Sigma

    return pref

class FINCOResults:
    """
    A class for FINCO results. The class allows reading results directly from
    datasets, or from a stored file

    The class parses the results file into the corresponding databases, loading
    them as necessary, and exposes convinience functions for results analysis.

    Parameters
    ----------
    file_path : string
        Path of the results file to load. Set to None in order to read from
        provided dataset.
    data : pandas.DataFrame
        Datasets with raw results to read from. Used only if file_path is None.
    gamma_f : float
        Width of the Gaussian projection on the real axis


    Notes
    -----
    The FINCO algorithm produces several result datasets, that can then be used to
    analyse the propagation results and reconstruct the wavefunction. The datasets
    are stored as pandas DataFrames, with the index of each entry being the the time
    step it was taken.

    It should be noted that this class accesses results files on demand, loading
    the required dataset as needed, and does not keep a copy. Therefore it is
    advised to retrieve the datasets as few as necessary, as reading files is
    inefficient.

    Datasets
    --------
    trajectories :
        Containing information about the trajectory propagation in time, for
        analysis and wavepacket reconstruction.
    projection_map :
        Containing information about the projection map of each  trajectory onto
        the real axis
    caustics :
        Containing information required for indentifying caustics and dealing
        with Stokes phenomenon.
    """

    data: pd.DataFrame
    file_path: str
    gamma_f: float

    def __init__(self, file_path: Union[str, None],
                 data: Union[pd.DataFrame, None], gamma_f: int):
        # Results
        self.file_path = file_path
        self.data = data

        # System
        self.gamma_f = gamma_f
        
        self._populate_data()

    def __len__(self):
        return self.nrows
        
    def __getattr__(self, name):
        try:
            print(f"I was called with a name {name}")
            if self.file_path is not None:
                with pd.HDFStore(path=self.file_path, mode='r', complevel=5) as file:
                    return file.get(key='results')[name]
            else:
                return self.data[name]
        except Exception as E:
            raise AttributeError(name) from E

    def __repr__(self):
        if self.file_path is not None:
            with pd.HDFStore(path=self.file_path, mode='r', complevel=5) as file:
                details = file.info()
            return f"FINCO results dataset at {self.file_path}. Details:\n{details}"

        return f"FINCO results dataset in memory with {self.data.size} entries"

    def _populate_data(self):
        if self.file_path is not None:
            with pd.HDFStore(path=self.file_path, mode='r', complevel=5) as file:
                desc = file.select(key='results', iterator=True, chunksize=1000)
                self.nrows = desc.s.nrows
                self.ncols = desc.s.ncols - len(desc.s.data_columns)
                self.size = self.nrows * self.ncols
                self.shape = (self.nrows, self.ncols)
        else:
            self.nrows = self.data.shape[0]
            self.ncols = self.data.shape[1]
            self.size = self.data.size
            self.shape = (self.nrows, self.ncols)
        
    def merge(self, other):
        """
        Merges two result datasets together. The user should make sure the are
        no index collisions.

        Parameters
        ----------
        other : FINCOResults
            Other results to merge.
        """
        if self.file_path is not None:
            with pd.HDFStore(path=self.file_path, mode='a', complevel=5) as file:
                file.append(key='results', value=other.get_results())
        else:
            self.data.append(other.get_results())
            
        self._populate_data()

    def get_results(self, start: Optional[int] = None,
                    end: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieves a slice of the raw results at given timesteps.

        Parameters
        ----------
        start : int, optional
            Starting timestep. If None, then the dataset is taken from the
            first timestep.
        end : int, optional
            Ending timestep (exclusive). If None, then the dataset is taken up
            to (and including) the last timestep.

        Returns
        -------
        dataset : pandas.DataFrame
            The requested raw results
        """
        if self.file_path is not None:
            with pd.HDFStore(path=self.file_path, mode='r', complevel=5) as file:
                if start is not None:
                    if end is not None:
                        dataset = file.select(key='results',
                                           where=f'(timestep >= {start}) & (timestep < {end})')
                    else:
                        dataset = file.select(key='results', where=f'timestep >= {start}')
                else:
                    if end is not None:
                        dataset = file.select(key='results', where=f'timestep < {end}')
                    else:
                        dataset = file.get(key='results')
        else:
            mask = np.ones_like(self.data.index)
            if start is not None:
                mask &= self.data.index.get_level_values(1) >= start
            if end is not None:
                mask &= self.data.index.get_level_values(1) < end

            dataset = self.data[mask.astype(bool)]

        return dataset.sort_index()

    def get_trajectories(self, start: Optional[int] = None,
                         end: Optional[int] = None, threshold: int = -1) -> pd.DataFrame:
        """
        Retrieves the trajectory data at given timestep.

        Parameters
        ----------
        start : int, optional
            Starting timestep. If None, then the dataset is taken from the
            first timestep.
        end : int, optional
            Ending timestep (exclusive). If None, then the dataset is taken up
            to (and including) the last timestep.
        threshold : float, optional
            Threshold for a trajectory's contribution. If positive, the function
            throws every trajectory whose Gaussian prefactor is more than threshold.
            The default is -1.

        Returns
        -------
        trajectories : pandas.DataFrame
            Trajectory dataset. Consists of the following fields:

            q0 : complex
                Initial location
            p0 : complex
                Initial momentum
            t : complex
                Time at timestep
            q : complex
                Location at timestep
            p : complex
                Momentum at timestep
            pref : complex
                Prefactor of the trajectory for its contribution to the
                integral, at timestep
        """
        results = self.get_results(start=start, end=end)
        pref = _calc_pref(results, self.gamma_f)

        if threshold > 0:
            results = results[np.abs(pref) < threshold]
            pref = pref[np.abs(pref) < threshold]

        return pd.DataFrame({'q0': results.q0, 'p0': results.p0,
                             't': results.t, 'q': results.q, 'p': results.p,
                             'pref': pref}, index=results.index)

    def get_projection_map(self, step: int) -> pd.DataFrame:
        """
        Calculates and returns a map of the Gaussian projection on the real
        axis, at given timestep. Used to deal with Stokes phenomena.

        Parameters
        ----------
        step : int
            Timestep to retrieve trajectories data for.

        Returns
        -------
        proj_map: pandas.DataFrame
            Dataset containing the projection map. Consists of the following
            fields:

            q0 : complex
                Initial location
            xi : complex
                The Gaussian projection map
            sigma : complex
                Exponent added to the Gaussian due to the projection
        """
        results = self.get_results(start=step, end=step+1)
        *_, xi, sigma = _calc_projection(results, self.gamma_f)

        return pd.DataFrame({'q0': results.q0, 'xi': xi,
                             'sigma': sigma}, index=results.index)

    def get_caustics_map(self, step: int) -> pd.DataFrame:
        """
        Calculates and returns a map used to determine where caustics are
        in the system, at given timestep. Used to deal with Stokes phenomena.

        Parameters
        ----------
        step : int
            Timestep to retrieve trajectories data for.

        Returns
        -------
        caustics_map: pandas.DataFrame
            Dataset containing the caustics map. Consists of the following
            fields:

            q0 : complex
                Initial location
            xi_1 : complex
                First derivative of the Gaussian projection map w.r.t. q0.
                Should be zero at the caustics
            sigma_1 : complex
                First derivative of the sigma parameter w.r.t. q0. Used to
                calculate the Stokes parameter and identify Stokes lines.
        """
        results = self.get_results(start=step, end=step+1)
        xi_1, sigma_1 = _calc_derivatives(results, self.gamma_f)

        return pd.DataFrame({'q0': results.q0, 'xi_1': xi_1,
                             'sigma_1': sigma_1}, index=results.index)

    def reconstruct_psi(self, x: ArrayLike, step: int, S_F: Optional[pd.Series] = None,
                        threshold: int = -1, n_jobs: int = 1) -> ArrayLike:
        """
        Reconstructs the wavefunction at given timestep

        Parameters
        ----------
        x : 1D ArrayLike of floats
            x positions to reconstruct the wavefunction for.
        step : int
            Timestep to reconstruct the wavefunction for.
        S_F : pandas.Series, optional
            List of Berry factors for the trajectories. If given, then the
            prefactor of each trajectory will be multiplied with its corresponding
            factor. The default is None.
        threshold : float, optional
            Threshold for a trajectory's contribution. If positive, the function
            throws every trajectory whose Gaussian prefactor is more than threshold.
            The default is -1.
        n_jobs : int, optional
            Number of parallel jobs to use for reconstruction. Dataset will be
            divided equally between jobs. The default is 1.

        Returns
        -------
        psi : 1D ArrayLike of complex in the shape of x
            The reconstructed wavefunction.
        """
        def area(points):
            """Calculates area in complex space from a list of complex points"""
            if len(points) == 0:
                return 0
            x, y = points.real, points.imag
            return np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1]*y[0] - x[0]*y[-1]) / 2

        def process(block):
            """Calculates the reconstruction of given block of points"""
            points, qf, pf, pref = block
            ordered = (points.groupby('point')
                       .apply(lambda x_: x_.to_numpy()[np.argsort(np.angle(x_), axis=None)]))
            areas = np.array(list(map(area, ordered)))[:, np.newaxis]

            pref = pref.to_numpy()[:, np.newaxis]

            return np.nansum((1 / 32 / self.gamma_f**3 / np.pi**3) ** 0.25 * pref *
                             gf(x, qf, pf, self.gamma_f) * areas / 3, axis=0)

        # Load results and updated indices according to created mesh
        results = self.get_results(start=step, end=step+1)
        mesh = Mesh(results)

        # Calculate all needed data
        qf, pf, *_ = _calc_projection(results, self.gamma_f)
        pref = _calc_pref(results, self.gamma_f)
        if S_F is not None:
            pref *= S_F

        if threshold > 0:
            pref[np.abs(pref) > threshold] = 0

        # Drop coplanar points (they are not a part of areas)
        qf.drop(mesh.tri.coplanar[:,0], inplace=True)
        pf.drop(mesh.tri.coplanar[:,0], inplace=True)
        pref.drop(mesh.tri.coplanar[:,0], inplace=True)

        # Calculate areas for each point
        neighbors = mesh.get_neighbors_value(results.q0)

        relative = (neighbors.q0 -
                     results.q0.take(mesh.points_to_mesh(neighbors.index.get_level_values('point'))).to_numpy())
        blocks = np.array_split(relative.index.get_level_values('point').unique().
                                astype(int).to_numpy(), n_jobs)
        ranges = [relative.loc[b_[0]:b_[-1]] for b_ in blocks]
        qfs = [qf.loc[b_[0]:b_[-1]] for b_ in blocks]
        pfs = [pf.loc[b_[0]:b_[-1]] for b_ in blocks]
        prefs = [pref.loc[b_[0]:b_[-1]] for b_ in blocks]

        return np.nansum(Parallel(n_jobs=n_jobs, verbose=10)([delayed(process)(x)
                                                  for x in zip(ranges, qfs, pfs, prefs)]), axis=0)

    def show_plots(self, x: ArrayLike, y0: float, y1: float, interval: int,
                   threshold: float = -1, skip: int = 1):
        """
        Helper function. Does a slideshow of the reconstructed wavepackets at
        each timestep.

        Parameters
        ----------
        x : 1D ArrayLike of floats
            x positions to reconstruct the wavepacket for.
        y0 : float
            Lower bound of y-axis for the slideshow
        y1 : float
            Upper bound of y-axis for the slideshow
        interval : int
            Sleep duration between steps, in seconds.
        threshold : float, optional
            Threshold for a trajectory's contribution. If positive, the function
            throws every trajectory whose Gaussian prefactor is more than threshold.
            The default is -1.
        skip : int, optional
            Frame skip. Shows the reconstructed wavepacket 1 out of 'skip' frames.
            The default is 1.

        """
        plt.figure()
        plt.ylim(y0, y1)
        psi, = plt.plot(x, np.abs(self.reconstruct_psi(x, 0, threshold)))
        t_max = np.max(self.get_results().index)
        for i in range(t_max - 1):
            plt.waitforbuttonpress(interval)
            if not i % skip:
                psi.set_ydata(np.abs(self.reconstruct_psi(x, i+1, threshold)))
                plt.draw()


def load_results(file_path: str, gamma_f: float = 1) -> FINCOResults:
    """
    Loads a FINCO results file into object. Should be used instead of
    directly calling FINCOResults.

    Parameters
    ----------
    file_path : string
        Path of the results file to load. Set to None in order to read from
        provided datasets.
    gamma_f : float, optional
        Width of the Gaussian projection on the real axis. The default is 1.

    Raises
    ------
    FileNotFoundError
        Raised if file_path points to a file that does not exist.

    Returns
    -------
    result: FINCOResults
        The loaded FINCO results object.

    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file {file_path} does not exist.")

    return FINCOResults(file_path, data=None, gamma_f=gamma_f)



def results_from_data(data: pd.DataFrame, gamma_f: float = 1) -> FINCOResults:
    """
    Creates a FINCO results object from dataset. Should be used instead of
    directly calling FINCOResults.

    Parameters
    ----------
    data : pandas.DataFrame
        dataset of trajcetories to wrap. Should have the same form as described
        in create_ics()
    gamma_f : float, optional
        Width of the Gaussian projection on the real axis. The default is 1.

    Returns
    -------
    result: FINCOResults
        The wrapped FINCO results object.

    """

    return FINCOResults(file_path=None, data=data, gamma_f=gamma_f)

def get_view(results: FINCOResults, start: Optional[int] = None,
             end: Optional[int] = None) -> FINCOResults:
    """
    Returns a view of a results dataset as a FINCOResults object. This is done
    by loading a subset of the results and creating a FINCOResults around it.

    As this is a view, data in the returned object should be treated as read-only.
    It does not change the results file.

    Parameters
    ----------
    results : FINCOResults
        Results object to return a view of.
    start : int, optional
        Starting timestep to take, as in FINCOResults.get_results(). The default is None.
    end : int, optional
        Ending timestep to take, as in FINCOResults.get_results(). The default is None.

    Returns
    -------
    view : FINCOResults
        The resulting view object
    """
    return results_from_data(data = results.get_results(start, end),
                             gamma_f = results.gamma_f)

class FINCOWriter:
    """
    A class for writing FINCO results.

    This class is used internally to write the results, and is here mainly
    to allow seperation between the algorithm and results writing.

    If a results file path is provided, the class opens the file on creation and
    closes it on deletion, making it fit to work in a 'with' statement. If not,
    then the datasets are kept in memory, and can be accessed through the data
    class member.

    Parameters
    ----------
    file_path : string
        Path of the results file to write to. If set to None, then the datasets
        are kept in memory.
    append : bool, optional
        Whether to append data to the results file or overwrite it.
        Set to True to append. The default is False.
    """

    def __init__(self, file_path: Optional[str], append: bool = False):
        if file_path is not None:
            mode = 'a' if append else 'w'
            self.results_file = pd.HDFStore(path=file_path, mode=mode, complevel=5)
        else:
            self.results_file = None
            self.data = pd.DataFrame()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Finalizes and closes the results file, adding index before closing.
        """
        if self.results_file is not None:
            for key in self.results_file.keys():
                self.results_file.create_table_index(key=key, optlevel=9, kind='full')

            self.results_file.close()

    def add_results(self, results: pd.DataFrame):
        """
        Adds result to the dataset, in the form of a pandas DataFrame as
        described in create_ics()

        Parameters
        ----------
        results : pandas.DataFrame
            The results to add to the dataset. Should be in the form as described in
            create_ics()
        """

        if self.results_file is not None:
            self.results_file.put(key='results', value=results, append=True,
                                  format='table', index=False)
        else:
            self.data = pd.concat([self.data, results])
