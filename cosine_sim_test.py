import numpy as np
import pytest

import rec_sys.cosine_sim as cs

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

excepted_output_b = 0.03117550829633092


@pytest.mark.parametrize(
    "input_vector_1, input_vector_2, expected_output",
    [(vec_x_a, vec_y_a, excepted_output_a), (vec_x_b, vec_y_b, excepted_output_b)],
)
def test_centered_cosine_sim(input_vector_1, input_vector_2, expected_output):
    assert (
        pytest.approx(cs.centered_cosine_sim(input_vector_1, input_vector_2))
        == expected_output
    )


@pytest.mark.parametrize(
    "vec_1, vec_2, expected",
    [(vec_x_a, vec_y_a, excepted_output_a), (vec_x_b, vec_y_b, excepted_output_b)],
)
def test_fast_centered_cosine_sim(vec_1, vec_2, expected):
    assert pytest.approx(cs.fast_centered_cosine_sim(vec_1, vec_2)) == expected
