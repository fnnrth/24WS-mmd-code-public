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


def rate_all_items_sparse(orig_utility_matrix, user_index, neighborhood_size):
    print(
        f"\n>>> CF computation for UM w/ shape: "
        + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n"
    )

    clean_utility_matrix = center_and_nan_to_zero_sparse(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    avg_rating_user = np.nanmean(orig_utility_matrix[:, user_index].toarray())
    # Compute the cosine similarity between the user and all other users
    similarities = fast_centered_cosine_sim_sparse(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(
            np.isnan(orig_utility_matrix[item_index, :].data) == False
        )[0]
        # From those, get indices of users with the highest similarity
        best_among_who_rated = users_who_rated[
            np.argsort(similarities[users_who_rated].toarray().flatten())
        ]
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[
            ~np.isnan(similarities[best_among_who_rated].toarray().flatten())
        ]
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            # Compute the rating of the item
            sum_similarities = np.sum(np.abs(similarities[best_among_who_rated].data))
            bawr_similarities = similarities[best_among_who_rated].toarray().flatten()
            bawr_ratings_item = (
                orig_utility_matrix[item_index, best_among_who_rated]
                .toarray()
                .flatten()
            )
            rating_of_item = (
                np.sum(bawr_similarities * bawr_ratings_item) / sum_similarities
            )
        else:
            rating_of_item = np.nan
        print(
            f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}"
        )
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


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
print(rate_all_items_sparse(matrix_sp, 0, 2))
