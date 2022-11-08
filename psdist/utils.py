import numpy as np
import pickle


def cov2corr(cov_mat):
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    corr_mat = np.linalg.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat


def copy_into_new_dim(a, shape, axis=-1, method='broadcast', copy=False):
    """Copy an array into one or more new dimensions.
    
    The 'broadcast' method is much faster since it works with views instead of copies. 
    See 'https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times'
    """
    if type(shape) in [int, np.int32, np.int64]:
        shape = (shape,)
    if method == 'repeat':
        for i in range(len(shape)):
            a = np.repeat(np.expand_dims(a, axis), shape[i], axis=axis)
        return a
    elif method == 'broadcast':
        if axis == 0:
            new_shape = shape + a.shape
        elif axis == -1:
            new_shape = a.shape + shape
        else:
            raise ValueError('Cannot yet handle axis != 0, -1.')
        for _ in range(len(shape)):
            a = np.expand_dims(a, axis)
        if copy:
            return np.broadcast_to(a, new_shape).copy()
        else:
            return np.broadcast_to(a, new_shape)
    return None


# The following three functions are from Tony Yu's blog 
# (https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR). 
# They allow saving/loading ragged arrays in .npz format. This is helpful when 
# tracking a bunch that is losing/gaining particles; e.g., during turn-by-turn
# injection.
def stack_ragged(arrays, axis=0):
    """Stacks list of arrays along first axis.
    
    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(a)[axis] for a in arrays]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(arrays, axis=axis)
    return stacked, idx


def save_stacked_array(filename, arrays, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(arrays, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)
    
    
def load_stacked_arrays(filename, axis=0):
    """"Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file['stacked_index']
    stacked = npz_file['stacked_array']
    return np.split(stacked, idx, axis=axis)