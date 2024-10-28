import numpy as np
import scipy.sparse as sp


def center_and_nan_to_zero_sparse(matrix: sp.csr_matrix, axis=0):
    """Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    column_means = sp.csr_matrix(
        [
            np.nanmean(
                matrix[:, col].toarray()
            )  # Convert only the current column to dense
            for col in range(matrix.shape[1])
        ]
    )
    column_means_matrix = sp.vstack([column_means] * matrix.shape[0])
    # Subtract the mean from each axis
    matrix_centered = matrix - column_means_matrix
    matrix_centered.data = np.nan_to_num(matrix_centered.data)
    return matrix_centered


def centered_cosine_sim_sparse(u: sp.csr_matrix, v: sp.csr_matrix):
    u_cen = center_and_nan_to_zero_sparse(u).T
    v_cen = center_and_nan_to_zero_sparse(v)
    dot = u_cen.dot(v_cen)
    norm_u = sp.linalg.norm(u_cen)
    norm_v = sp.linalg.norm(v_cen)
    centered_cosine = dot.data[0] / (norm_u * norm_v)
    return centered_cosine


def fast_centered_cosine_sim_sparse(
    matrix: sp.csr_matrix, vector: sp.csr_matrix, axis=0
):
    matrix_cen = center_and_nan_to_zero_sparse(matrix, axis=axis)
    vector_cen = center_and_nan_to_zero_sparse(vector)
    norms = sp.linalg.norm(matrix_cen, axis=axis)
    norm_vector = sp.linalg.norm(vector_cen)
    matrix_cen = matrix_cen / norms
    dot = (matrix_cen.T).dot(vector_cen)
    scaled = dot / norm_vector
    return scaled
