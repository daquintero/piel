import jax.numpy as jnp
import time
import numpy as np


def unitary_permanent(
    unitary_matrix: jnp.ndarray,
) -> tuple:
    """
    The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

    ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

    Note that this function needs to be as optimised as possible, so we need to minimise our computational complexity of our operation.

    # TODO implement validation
    # TODO maybe implement subroutine if computation is taking forever.
    # TODO why two outputs? Understand this properly later.

    Args:
        unitary_permanent (np.ndarray): The unitary matrix.

    Returns:
        tuple: The circuit permanent and the time it took to compute it.

    """
    import thewalrus

    start_time = time.time()
    unitary_matrix_numpy = np.asarray(unitary_matrix)
    circuit_permanent = thewalrus.perm(unitary_matrix_numpy)
    end_time = time.time()
    computed_time = end_time - start_time
    return circuit_permanent, computed_time
