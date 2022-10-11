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


def project1d_contour(f, axis=0, level=0.1, fpr=None, normalize=True, return_frac=False):
    axis_proj = [i for i in range(f.ndim) if i != axis]
    if fpr is None:
        fpr = project(f, axis_proj)
    fpr = fpr / np.max(fpr)
    idx = np.where(fpr > level)
    frac = np.sum(fpr[idx]) / np.sum(fpr)
    idx = make_slice(f.ndim, axis_proj, idx)    
    p = np.sum(f[idx], axis=int(axis == 0))
    if normalize:
        p = p / np.sum(p)
    if return_frac:
        return p, frac
    return p


def project2d_contour(f, level=0.1, fpr=None, normalize=True, return_frac=False, axis=(2, 3)):
    # Compute the 3D mask.
    axis_proj = [i for i in range(f.ndim) if i not in axis]
    if fpr is None:
        fpr = utils.project(f, axis_proj)
    fpr = fpr / np.max(fpr)
    mask = fpr < level
    frac = np.sum(fpr[~mask]) / np.sum(fpr)

    # Copy the 3D mask into the two projected dimensions.
    mask = utils.copy_into_new_dim(mask, (f.shape[axis[0]], f.shape[axis[1]]), axis=-1, copy=True)
    # Put the dimensions in the correct order.        
    isort = np.argsort(list(axis_proj) + list(axis))
    mask = np.moveaxis(mask, isort, np.arange(5))
    
    # Project the masked 5D array onto the specified axis.    
    im = utils.project(np.ma.masked_array(f, mask=mask), axis=axis)
    if return_frac:
        return im, frac
    return im


def get_radii(coords, Sigma):
    COORDS = np.meshgrid(*coords, indexing='ij')
    shape = tuple([len(c) for c in coords])
    R = np.zeros(shape)
    Sigma_inv = np.linalg.inv(Sigma)
    for ii in tqdm(np.ndindex(shape)):
        vec = np.array([C[ii] for C in COORDS])
        R[ii] = np.sqrt(np.linalg.multi_dot([vec.T, Sigma_inv, vec]))
    return R


def radial_density(f, R, radii, dr=None):
    if dr is None:
        dr = 0.5 * np.max(R) / (len(R) - 1)
    fr = []
    for r in tqdm(radii):
        f_masked = np.ma.masked_where(np.logical_or(R < r, R > r + dr), f)
        # mean density within this shell...
        fr.append(np.mean(f_masked))
    return np.array(fr)