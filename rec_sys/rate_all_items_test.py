import numpy as np
import pytest
import scipy.sparse as sp

import rec_sys.cf_algorithms as ca
import rec_sys.cf_algorithms_sparse as cs

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
matrix_sp = sp.csr_matrix(matrix)


@pytest.mark.parametrize(
    "um_matrix, um_matrix_sp, user_index, neighborhood_size", [(matrix, 0, 2)]
)
def test_rate_all_items(um_matrix, um_matrix_sp, user_index, neighborhood_size):
    excepted = ca.rate_all_items(um_matrix, user_index, neighborhood_size)
    output = cs.rate_all_items_sparse(um_matrix_sp, user_index, neighborhood_size)
    assert np.all(excepted == output)


print(test_rate_all_items(matrix, matrix_sp, 0, 2))
