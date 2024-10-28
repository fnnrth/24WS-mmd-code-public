import numpy as np
import pytest
import scipy.sparse as sp

import rec_sys.cf_algorithms_sparse as cs
import rec_sys.cf_algorithms as cfa

# ---TEST CENTERED COSINE SIM SPRASE---
# Vectors for b.1)
k = 100
vec_x_a = np.arange(0, k)
vec_y_a = vec_x_a[::-1]

sp_x_a = sp.csr_matrix(vec_x_a).T
sp_y_a = sp.csr_matrix(vec_y_a).T

# Vectors for b.2)
c_values = [2, 3, 4, 5, 6]
b_list = [
    np.nan if any((i - c) % 10 == 0 for c in c_values) else i + 1 for i in range(k)
]
vec_x_b = np.array(b_list)
vec_y_b = vec_x_b[::-1]
sp_x_b = sp.csr_matrix(vec_x_b).T
sp_y_b = sp.csr_matrix(vec_y_b).T


@pytest.mark.parametrize(
    "sp_x, sp_y, vec_x, vec_y",
    [(sp_x_a, sp_y_a, vec_x_a, vec_y_a), (sp_x_b, sp_y_b, vec_x_b, vec_y_b)],
)
def test_centered_cosine_sim(sp_x, sp_y, vec_x, vec_y):
    computed_output = pytest.approx(cs.centered_cosine_sim_sparse(sp_x, sp_y))
    vec_x_cen = cfa.center_and_nan_to_zero(vec_x)
    vec_y_cen = cfa.center_and_nan_to_zero(vec_y)
    expected_output = cfa.cosine_sim(vec_x_cen, vec_y_cen)
    assert computed_output == expected_output


# print(test_centered_cosine_sim(sp_x_a, sp_y_a, vec_x_a, vec_y_a))

# ---TEST CENTERED FAST COSINE SIM SPRASE---
matrix = np.asarray(
    [
        [1.0, np.nan, 3.0, np.nan, np.nan, 5.0],
        [np.nan, np.nan, 5.0, 4.0, np.nan, np.nan],
        [2.0, 4.0, np.nan, 1.0, 2.0, np.nan],
        [np.nan, 2.0, 4.0, np.nan, 5.0, np.nan],
        [np.nan, np.nan, 4.0, 3.0, 4.0, 2.0],
        [1.0, np.nan, 3.0, np.nan, 3.0, np.nan],
    ]
)
vec = matrix[:, 0]
vec_sp = sp.csr_matrix(vec).T
matrix_sp = sp.csr_matrix(matrix)

cen_matrix = cfa.center_and_nan_to_zero(matrix)
cen_sparse = cs.center_and_nan_to_zero_sparse(matrix_sp)
cen_vec = cfa.center_and_nan_to_zero(vec)
expected_fast = cfa.fast_cosine_sim(cen_matrix, cen_vec, axis=0)


@pytest.mark.parametrize(
    "matrix_sp, vec_sp, matrix, vec", [(matrix_sp, vec_sp, matrix, vec)]
)
def test_fast_centered_cosine_sim(matrix_sp, vec_sp, matrix, vec):
    output = pytest.approx(
        (
            cs.fast_centered_cosine_sim_sparse(matrix_sp, vec_sp, axis=0)
            .toarray()
            .flatten()
        )
    )
    cen_matrix = cfa.center_and_nan_to_zero(matrix)
    cen_vec = cfa.center_and_nan_to_zero(vec)
    expected = cfa.fast_cosine_sim(cen_matrix, cen_vec, axis=0)
    assert np.all(output == expected)


# print(test_fast_centered_cosine_sim(matrix_sp, vec_sp, matrix, vec))
