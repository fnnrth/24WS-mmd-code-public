import numpy as np
import pytest
import scipy.sparse as sp

import rec_sys.cosine_sim as cs

# Vectors for b.1)
k = 100
vec_x_a = np.arange(0, k)

sp_x_a = sp.csr_matrix(vec_x_a)
vec_y_a = vec_x_a[::-1]
sp_y_a = sp.csr_matrix(vec_y_a)

excepted_output_a = -0.9696969696969697
# Vectors for b.2)
c_values = [2, 3, 4, 5, 6]
b_list = [
    np.nan if any((i - c) % 10 == 0 for c in c_values) else i + 1 for i in range(k)
]
vec_x_b = np.array(b_list)

sp_x_b = sp.csr_matrix(vec_x_b)
vec_y_b = vec_x_b[::-1]
sp_y_b = sp.csr_matrix(vec_y_b)

excepted_output_b = -0.8019070321811681


@pytest.mark.parametrize(
    "input_vector_1, input_vector_2, expected_output",
    [(sp_x_a, sp_y_a, excepted_output_a), (sp_x_b, sp_y_b, excepted_output_b)],
)
def test_centered_cosine_sim(input_vector_1, input_vector_2, expected_output):
    computed_output = pytest.approx(
        cs.centered_cosine_sim(input_vector_1, input_vector_2)
    )
    assert computed_output == expected_output


"""
@pytest.mark.parametrize(
    "vec_1, vec_2, expected",
    [(vec_x_a, vec_y_a, excepted_output_a), (vec_x_b, vec_y_b, excepted_output_b)],
)
def test_fast_centered_cosine_sim(vec_1, vec_2, expected):
    assert pytest.approx(cs.fast_centered_cosine_sim(vec_1, vec_2)) == expected
"""
