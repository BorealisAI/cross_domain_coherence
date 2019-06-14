import numpy as np

def random_permutation_matrix(N, dtype=np.float32):
    """
    Generate a random permutation matrix.

    :param N: dimension of the permutation matrix
    :return: a numpy array with shape (N, N)
    """
    A = np.identity(N, dtype=dtype)
    idx = np.random.permutation(N)
    return A[idx, :]

def generate_random_pmatrices(N, size):
    """
    Generate a batch of random permutation matrices.

    :param N: dimension of the permutation matrices
    :param size: number of generated matrices
    :return: a numpy array with shape (size, N, N)
    """
    res = []
    for i in range(size):
        res.append(random_permutation_matrix(N))
    return np.array(res)
