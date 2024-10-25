import numpy as np
import scipy.sparse as sp


def centered_cosine_sim(vec_1, vec_2, axis=0):
    """Compute the cosine similarity between the matrix and the vector"""
    # Center the matrix
    vec_1_centered = vec_1 - np.nanmean(vec_1, axis=axis)
    vec_2_centered = vec_1 - np.nanmean(vec_2, axis=axis)
    # Compute the cosine similarity
    cosine_sim = np.dot(vec_1_centered, vec_2_centered) / (
        np.linalg.norm(vec_1_centered) * np.linalg.norm(vec_2_centered)
    )
    return cosine_sim


def fast_centered_cosine_sim(vec_1, vec_2, axis=0):
    # Convert the vectors to sparse arrays
    vec_1_sp = sp.csr_array(vec_1)
    vec_2_sp = sp.csr_array(vec_2)
    # Center the sparse arrays
    vec_1_sp -= vec_1_sp.sum() / vec_1_sp.shape[0]
    vec_2_sp -= vec_2_sp.sum() / vec_2_sp.shape[0]
    # Compute the cosine similarity
    dot_product = vec_1_sp.dot(vec_2_sp)
    norm_vec_1 = sp.linalg.norm(vec_1_sp)
    norm_vec_2 = sp.linalg.norm(vec_2_sp)
    cosine_sim = dot_product / (norm_vec_1 * norm_vec_2)
    return cosine_sim
