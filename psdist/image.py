"""N-dimensional images."""
import numpy as np


def make_slice(n, axis=0, ind=0):
    """Return a slice index array.
    
    n : int
        The length of the slice index. 
    axis : list[int]
        The sliced axes.
    ind : list[int] or list[tuple]
        The indices along the sliced axes. If a tuple is provided, this
        defines the (min, max) index.
        
    Returns
    -------
    idx : tuple
        The slice index array.
    """
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    idx = n * [slice(None)]
    for k, i in zip(axis, ind):
        if i is None:
            continue
        elif type(i) is tuple and len(i) == 2:
            idx[k] = slice(i[0], i[1])
        else:
            idx[k] = i
    return tuple(idx)


def project(image, axis=0):
    """Project image onto one or more axes.
    
    Parameters
    ----------
    image : ndarray
        An n-dimensional image.
    axis : list[int]
        The axes onto which the image is projected, i.e., the
        axes which are not summed over. Can be an int or list
        or ints.
    
    Returns
    -------
    proj : ndarray
        The projection of `image` onto the specified axis.
    """
    # Sum over specified axes.
    n = image.ndim
    if type(axis) is int:
        axis = [axis]
    axis = tuple(axis)
    axis_sum = tuple([i for i in range(image.ndim) if i not in axis])
    proj = np.sum(image, axis_sum)
    
    # Order the remaining axes.
    n = proj.ndim
    loc = list(range(n))
    destination = np.zeros(n, dtype=int)
    for i, index in enumerate(np.argsort(axis)):
        destination[index] = i
    for i in range(n):
        if loc[i] != destination[i]:
            j = loc.index(destination[i])
            proj = np.swapaxes(proj, i, j)
            loc[i], loc[j] = loc[j], loc[i]
    return proj