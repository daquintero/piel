import sax
import time
import thewalrus
import jax.numpy as jnp
from ..tools.sax import sax_to_s_parameters_standard_matrix
from typing import Optional
import numpy as np

__all__ = ["sax_circuit_permanent", "subunitary_selection", "unitary_permanent"]


def sax_circuit_permanent(
    sax_input: sax.SType,
) -> tuple:
    """
    The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

    ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

    # TODO maybe implement subroutine if computation is taking forever.

    Args:
        sax_input (sax.SType): The sax S-parameter dictionary.

    Returns:
        tuple: The circuit permanent and the time it took to compute it.

    """
    s_parameter_standard_matrix, port_order = sax_to_s_parameters_standard_matrix(
        sax_input
    )
    circuit_permanent, computed_time = unitary_permanent(s_parameter_standard_matrix)
    return circuit_permanent, computed_time


def subunitary_selection(
    unitary_matrix: jnp.ndarray,
    stop_index: tuple,
    start_index: Optional[tuple] = (0, 0),
):
    """
    This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
    the output matrix is also a unitary.

    TODO implement validation of a 2D matrix.
    """
    start_row = start_index[0]
    end_row = stop_index[0]
    start_column = start_index[1]
    end_column = stop_index[1]
    column_range = jnp.arange(start_column, end_column + 1)
    row_range = jnp.arange(start_row, end_row + 1)
    unitary_matrix_row_selection = unitary_matrix.at[row_range, :].get()
    unitary_matrix_row_column_selection = unitary_matrix_row_selection.at[
        :, column_range
    ].get()
    return unitary_matrix_row_column_selection


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
    start_time = time.time()
    unitary_matrix_numpy = np.asarray(unitary_matrix)
    circuit_permanent = thewalrus.perm(unitary_matrix_numpy)
    end_time = time.time()
    computed_time = end_time - start_time
    return circuit_permanent, computed_time
