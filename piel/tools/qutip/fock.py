from itertools import product
import math
import numpy as np
import jax.numpy as jnp
from typing import Optional, Literal
import qutip

__all__ = [
    "all_fock_states_from_photon_number",
    "convert_qobj_to_jax",
    "convert_output_type",
    "fock_state_nonzero_indexes",
    "fock_state_to_photon_number_factorial",
    "fock_states_at_mode_index",
]


def all_fock_states_from_photon_number(
    mode_amount: int,
    photon_amount: int = 1,
    output_type: Literal["qutip", "jax"] = "qutip",
) -> list:
    """
    For a specific amount of modes, we can generate all the possible Fock states for whatever amount of input photons we desire. This returns a list of all corresponding Fock states.

    Args:
        mode_amount (int): The amount of modes in the system.
        photon_amount (int, optional): The amount of photons in the system. Defaults to 1.
        output_type (str, optional): The type of output. Defaults to "qutip".

    Returns:
        list: A list of all the Fock states.
    """
    photon_numbers = [[i] for i in range(photon_amount + 1)]
    states = []
    for photon_number in product(photon_numbers, repeat=mode_amount):
        state_values = np.array(photon_number).reshape(mode_amount, 1)
        state = convert_output_type(state_values, output_type)
        states.append(state)
    return states


def convert_qobj_to_jax(qobj: qutip.Qobj) -> jnp.ndarray:
    return jnp.array(qobj.data.todense())


def convert_output_type(array: np.ndarray, output_type: Literal["qutip", "jax"]):
    if output_type == "qutip":
        array = qutip.Qobj(array)
    elif output_type == "jax":
        array = jnp.array(array)
    else:
        raise ValueError("The output type must be either 'qutip' or 'jax'.")
    return array


def fock_state_to_photon_number_factorial(
    fock_state: qutip.Qobj | jnp.ndarray,
) -> float:
    """
    This function converts a Fock state defined as:

    .. math::
        \newcommand{\ket}[1]{\left|{#1}\right\rangle}
        \ket{f_1} = \ket{j_1, j_2, ... j_N}$

    and returns:

    .. math::

        j_1^{'}! j_2^{'}! ... j_N^{'}!

    Args:
        fock_state (qutip.Qobj): A QuTip QObj representation of the Fock state.

    Returns:
        float: The photon number factorial of the Fock state.
    """
    # TODO implement checks of Fock state validity
    if isinstance(fock_state, qutip.Qobj):
        fock_state = convert_qobj_to_jax(fock_state)

    photon_number_factorial = 1
    for photon_number in fock_state:
        photon_number_amount = photon_number[0].real
        photon_number_factorial *= math.factorial(int(photon_number_amount))
    return photon_number_factorial


def fock_state_nonzero_indexes(fock_state: qutip.Qobj | jnp.ndarray) -> tuple[int]:
    """
    This function returns the indexes of the nonzero elements of a Fock state.

    Args:
        fock_state (qutip.Qobj): A QuTip QObj representation of the Fock state.

    Returns:
        tuple: The indexes of the nonzero elements of the Fock state.
    """
    # TODO implement checks of Fock state validity
    if isinstance(fock_state, qutip.Qobj):
        fock_state = convert_qobj_to_jax(fock_state)

    nonzero_indexes = []
    for index, photon_number in enumerate(fock_state):
        photon_number_amount = photon_number[0].real
        if photon_number_amount != 0:
            nonzero_indexes.append(index)
    return tuple(nonzero_indexes)


def fock_states_at_mode_index(
    mode_amount: int,
    target_mode_index: int,
    maximum_photon_amount: Optional[int] = 1,
    output_type: Literal["qutip", "jax"] = "qutip",
) -> list:
    """
    This function returns a list of valid Fock states that fulfill a condition of having a maximum photon number at a specific mode index.

    Args:
        mode_amount (int): The amount of modes in the system.
        target_mode_index (int): The mode index to check the photon number at.
        maximum_photon_amount (int, optional): The amount of photons in the system. Defaults to 1.
        output_type (str, optional): The type of output. Defaults to "qutip".

    Returns:
        list: A list of all the Fock states.
    """

    def check_photon_number_at_mode(state_value: np.ndarray) -> bool:
        # Check if mode_index is valid
        if target_mode_index < len(state_value):
            # Return comparison result
            return (state_value[target_mode_index][0] <= maximum_photon_amount) and (
                state_value[target_mode_index][0] > 0
            )
        else:
            # Index out of range.
            return False

    photon_numbers = [[i] for i in range(maximum_photon_amount + 1)]
    states = []

    for photon_number in product(photon_numbers, repeat=mode_amount):
        state_values = np.array(photon_number).reshape(mode_amount, 1)
        if check_photon_number_at_mode(state_values):
            state = convert_output_type(state_values, output_type)
            states.append(state)
    return states
