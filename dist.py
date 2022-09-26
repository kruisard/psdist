"""Particle distributions (point clouds)."""
import numpy as np


def radial_extent(X, fraction=1.0):
    """Return radius of sphere containing fraction of points.

    X : n-d array
        Coordinate array.
    fraction : float
        Fraction of points in sphere.
    """
    radii = np.linalg.norm(X, axis=0)
    radii = np.sort(radii)
    if (X.shape[0] * fraction < 1.0):
        imax = X.shape[0] - 1
    else:
        imax = int(np.round(X.shape[0] * fraction))
    try:
        radius = radii[imax]
    except:
        radius = 0.0
    return radius


def slice_box(X, axis=None, limits=None):
    """Return points within box.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    limits : list[tuple]
        List of (min, max) pairs defining the edges of a box.
        
    Returns
    -------
    ndarray, shape (m, d)
        Points within the box.
    """
    n = X.shape[1]
    if axis is None:
        axis = tuple(range(n))
    if type(axis) is not tuple:
        axis = (axis,)
    if limits is None:
        limits = n * [(-np.inf, np.inf)]
    if type(limits) is not list:
        limits = [limits]
    conditions = []
    for i, (umin, umax) in zip(axis, limits):
        conditions.append(X[:, i] > umin)
        conditions.append(X[:, i] < umax)
    idx = np.logical_and.reduce(conditions)
    return X[idx]


def slice_sphere(X, axis=0, r=None):
    """Return points within sphere.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    r : float
        Radius of sphere.

    Returns
    -------
    ndarray, shape (m, d)
        Points within boundary.
    """
    n = X.shape[1]
    if axis is None:
        axis = tuple(range(n))
    if r is None:
        r = np.inf
    radii = np.linalg.norm(X[:, axis], axis=0)
    idx = radii < r
    return X[idx]


def slice_ellipsoid(X, axis=0, limits=None):
    """Return points within ellipsoid.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    limits : list[float]
        Semi-axes of ellipsoid.

    Returns
    -------
    ndarray, shape (m, d)
        Points within boundary.
    """
    n = X.shape[1]
    if axis is None:
        axis = tuple(range(n))
    if limits is None:
        limits = n * [np.inf]
    limits = np.array(limits)
    radii = np.sum((X[:, axis] / (0.5 * limits))**2, axis=1)
    idx = radii < 1.0
    return X[idx]


def histogram_bin_edges(X, bins=None, binrange=None):
    """Multi-dimensional histogram bin edges."""
    if bins is None:
        bins = 10
    if type(bins) is not list:
        bins = X.shape[1] * [bins]
    if type(binrange) is not list:
        binrange = X.shape[1] * [binrange] 
    edges = [np.histogram_bin_edges(X[:, i], bins[i], binrange[i]) 
             for i in range(X.shape[1])]
    
    
def histogram(X, bins=None, binrange=None):
    """Multi-dimensional histogram."""
    edges = histogram_bin_edges(X, bins=None, binrange=None)
    return np.histogramdd(X, bins=edges)