import numpy as np
import pickle


def save_pickle(filename, item):
    """Convenience function to save pickled file."""
    with open(filename, 'wb') as file:
        pickle.dump(item, file)
        
        
def load_pickle(filename):
    """Convenience function to load pickled file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def cov2corr(cov_mat):
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    corr_mat = np.linalg.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat


def symmetrize(a):
    """Return symmetrized version of upper or lower triangular matrix `a`."""
    return a + a.T - np.diag(a.diagonal())


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])


def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda v: np.matmul(M, v), 1, X)


def max_indices(a):
    """Return indices of maximum element in array."""
    return np.unravel_index(np.argmax(a), a.shape)    


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


def permutations_with_replacement(elements, n):
    """Return unique permutations of elements.
    
    https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    """
    def permutations_helper(elements, result_list, d):
        if d < 0 :
            yield tuple(result_list)
        else:
            for element in elements:
                result_list[d] = element
                for g in permutations_helper(elements, result_list, d - 1):
                    yield g
                    
    return permutations_helper(elements, [0] * n, n - 1)


def multiset_permutations(elements):
    from sympy.utilities.iterables import multiset_permutations
    return multiset_permutations(elements)