import numpy as np
import scipy.sparse as sp


def center_and_nan_to_zero_sparse(matrix: sp.csr_matrix, axis=0):
    """Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix.data, axis=axis)
    # Subtract the mean from each axis
    matrix.data = matrix.data - means
    matrix.data = np.nan_to_num(matrix.data)
    return matrix


def centered_cosine_sim_sparse(u: sp.csr_matrix, v: sp.csr_matrix):
    dot = u.dot(v.T).data[0]
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
    cen_u = center_and_nan_to_zero_sparse(u)
    cen_v = center_and_nan_to_zero_sparse(v)
    return centered_cosine_sim_sparse(cen_u, cen_v)


def fast_centered_cosine_sim(matrix, vector, axis=0):
    """Compute the cosine similarity between two vectors using sparse matrices"""
    cen_vec = center_and_nan_to_zero_sparse(vector)
    cen_matrix = center_and_nan_to_zero_sparse(matrix, axis=axis)
    return fast_cosine_sim_sparse(cen_matrix, cen_vec, axis=axis)
