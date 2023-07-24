import jax.numpy as jnp
import numpy as np
from typing import Optional
import qutip  # NOQA : F401

__all__ = [
    "standard_s_parameters_to_qutip_qobj",
    "verify_matrix_is_unitary",
    "subunitary_selection_on_range",
    "subunitary_selection_on_index",
]


def matrix_to_qutip_qobj(
    s_parameters_standard_matrix: jnp.ndarray,
):
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
    Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
    already in described in piel/piel/sax/utils.py.

    From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly. https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

    For example, a ``qutip`` representation of an s-gate gate would be:

    ..code-block::

        import numpy as np
        import qutip
        # S-Gate
        s_gate_matrix = np.array([[1.,   0], [0., 1.j]])
        s_gate = qutip.Qobj(mat, dims=[[2], [2]])

    In mathematical notation, this S-gate would be written as:

    ..math::

        S = \\begin{bmatrix}
            1 & 0 \\\\
            0 & i \\\\
        \\end{bmatrix}

    Args:
        s_parameters_standard_matrix (nso.ndarray): A dictionary of S-parameters in the form of a SDict from `sax`.

    Returns:
        qobj_unitary (qutip.Qobj): A QuTip QObj representation of the S-parameters in a unitary matrix.

    """
    s_parameter_standard_matrix_numpy = np.asarray(s_parameters_standard_matrix)
    qobj_unitary = qutip.Qobj(s_parameter_standard_matrix_numpy)
    return qobj_unitary


def subunitary_selection_on_index(
    unitary_matrix: jnp.ndarray,
    rows_index: jnp.ndarray | tuple,
    columns_index: jnp.ndarray | tuple,
):
    """
    This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
    the output matrix is also a unitary.

    TODO implement validation of a 2D matrix.
    """
    if type(rows_index) is tuple:
        rows_index = jnp.asarray(rows_index)

    if type(columns_index) is tuple:
        rows_index = jnp.asarray(columns_index)

    unitary_matrix_row_selection = unitary_matrix.at[rows_index, :].get()
    unitary_matrix_row_column_selection = unitary_matrix_row_selection.at[
        :, columns_index
    ].get()
    return unitary_matrix_row_column_selection


def subunitary_selection_on_range(
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
    unitary_matrix_row_column_selection = subunitary_selection_on_index(
        unitary_matrix=unitary_matrix,
        rows_index=row_range,
        columns_index=column_range,
    )
    return unitary_matrix_row_column_selection


def verify_matrix_is_unitary(matrix: jnp.ndarray) -> bool:
    """
    Verify that the matrix is unitary.

    Args:
        matrix (jnp.ndarray): The matrix to verify.

    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    qobj = matrix_to_qutip_qobj(matrix)
    return qobj.check_isunitary()


standard_s_parameters_to_qutip_qobj = matrix_to_qutip_qobj
