import jax.numpy as jnp
from jax.experimental import sparse
import jax
import numpy.random as random
import numpy as np


def _insert_max_degrees(t: int, max_list: list[int], coords: list) -> list[int]:
    for max_idx, max_val in enumerate(max_list):
        coeff_idx = random.randint(0, t)
        coords[coeff_idx][max_idx] = max_val
    return coords


def _insert_new_coords(new_coords_list, max_list):
    for idx_of_coord, coord in enumerate(new_coords_list):
        if coord is None:
            new_coords_list[idx_of_coord] = random.randint(0, max_list[idx_of_coord])
    return new_coords_list


def _sample_coeff(max_val: int):
    return random.randint(0, max_val)


def generate_random_polynomial(t: int, max_list: list[int], max_coeff):
    coords = [[None for i in max_list] for i in range(t)]
    coords = _insert_max_degrees(t, max_list, coords)
    data = [None for i in range(t)]

    for idx_in_coords_list, coords_list in enumerate(coords):
        # Each iteration a random pair of coordinates and a coefficient is generated
        while True:
            if all(x is not None for x in coords_list):
                data[idx_in_coords_list] = _sample_coeff(max_coeff)
                break
            new_coords_list = _insert_new_coords(coords_list.copy(), max_list)
            if new_coords_list not in coords:
                coords[idx_in_coords_list] = new_coords_list
                data[idx_in_coords_list] = _sample_coeff(max_coeff)
                break

    coords_jp = jnp.array(coords)
    data_jp = jnp.array(data)
    matrix = sparse.BCOO((data_jp, coords_jp), shape=(i for i in max_list))
    return matrix


def eval_polynomial(matrix: sparse.BCOO, input: jnp.array):
    exp = input**matrix.indices
    prod = jnp.prod(exp, axis=1)
    result = jnp.sum(prod * matrix.data)
    return result


def eval_polynomial_vectorized(matrix: sparse.BCOO, input: jnp.array):
    eval = jax.vmap(lambda x: eval_polynomial(matrix, x), in_axes=0)(input)
    return eval


def generate_training_set(
    polynomial_matrix, n_points, start_samples, end_samples, noise_frac, rnd_seed
):
    x = jnp.linspace(start_samples, end_samples, n_points)
    y_pure = eval_polynomial_vectorized(polynomial_matrix, x)
    # Add some noise to data
    rnd_key = jax.random.PRNGKey(rnd_seed)
    y_with_noise = y_pure + y_pure * noise_frac * jax.random.normal(
        rnd_key, (n_points,)
    )
    return x, y_pure, y_with_noise


start_samples = np.array([-1, -1, -1])
end_samples = np.array([1, 1, 1])
polynomial = generate_random_polynomial(3, [3, 3, 3], 10)
x = generate_training_set(polynomial, 100, start_samples, end_samples, 0.25, 42)
print(x)
