import numpy as np
import scipy.sparse as sp
from rec_sys.cf_algorithms_to_complete import (
    center_and_nan_to_zero,
    cosine_sim,
    fast_cosine_sim,
)


def center_and_nan_to_zero_sparse(matrix: sp.csr_matrix, axis=0):
    """Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix.data, axis=axis)
    # Subtract the mean from each axis
    matrix.data = matrix.data - means
    matrix.data = np.nan_to_num(matrix.data)
    return matrix


def cosine_sim_sparse(u: sp.csr_matrix, v: sp.csr_matrix):
    dot = u.dot(v)
    norm_u = sp.linalg.norm(u)
    norm_v = sp.linalg.norm(v)
    return dot / (norm_u * norm_v)


def fast_cosine_sim_sparse(matrix: sp.csr_matrix, vector: sp.csr_matrix, axis=0):
    dot = matrix.dot(vector)
    norms = sp.linalg.norm(matrix, axis=axis)
    norm_vector = sp.linalg.norm(vector)
    matrix.data = matrix.data / norms
    scaled = dot / norm_vector
    return scaled


def centered_cosine_sim(u, v):
    """Compute the cosine similarity between two vectors"""
    cen_u = center_and_nan_to_zero(u)
    cen_v = center_and_nan_to_zero(v)
    return cosine_sim(cen_u, cen_v)


def fast_centered_cosine_sim(matrix, vector, axis=0):
    """Compute the cosine similarity between two vectors using sparse matrices"""
    cen_vec = center_and_nan_to_zero(vector)
    cen_matrix = center_and_nan_to_zero_sparse(matrix, axis=axis)
    return fast_cosine_sim(cen_matrix, cen_vec, axis=axis)


# Vectors for b.1)
k = 100
vec_x_a = np.arange(1, 101)
vec_y_a = np.array([vec_x_a[k - 1 - i] for i in range(k)])

excepted_output_a = -1
# Vectors for b.2)
c_values = [2, 3, 4, 5, 6]
vec_x_b = np.array(
    [np.nan if any((i - c) % 10 == 0 for c in c_values) else i + 1 for i in range(100)]
)
vec_y_b = np.array([vec_x_b[k - 1 - i] for i in range(k)])
centered_cosine_sim(vec_x_a, vec_y_a)
fast_centered_cosine_sim(vec_x_b, vec_y_b)
