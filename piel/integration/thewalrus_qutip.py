import jax.numpy as jnp
import qutip

from ..tools.thewalrus import unitary_permanent
from ..tools.qutip import (
    fock_state_nonzero_indexes,
    fock_state_to_photon_number_factorial,
    subunitary_selection_on_index,
)


def fock_transition_probability_amplitude(
    initial_fock_state: qutip.Qobj | jnp.ndarray,
    final_fock_state: qutip.Qobj | jnp.ndarray,
    unitary_matrix: jnp.ndarray,
):
    """
    This function returns the transition probability amplitude between two Fock states when propagating in between
    the unitary_matrix which represents a quantum state circuit.

    Note that based on (TODO cite Jeremy), the initial Fock state corresponds to the columns of the unitary and the
    final Fock states corresponds to the rows of the unitary.

    .. math ::
        \newcommand{\ket}[1]{\left|{#1}\right\rangle}

    The subunitary :math:`U_{f_1}^{f_2}` is composed from the larger unitary by selecting the rows from the output state
    Fock state occupation of :math:`\ket{f_2}`, and columns from the input :math:`\ket{f_1}`. In our case, we need to select the
    columns indexes :math:`(0,3)` and rows indexes :math:`(1,2)`.

    If we consider a photon number of more than one for the transition Fock states, then the Permanent needs to be
    normalised. The probability amplitude for the transition is described as:

    .. math ::
        a(\ket{f_1} \to \ket{f_2}) = \frac{\text{per}(U_{f_1}^{f_2})}{\sqrt{(j_1! j_2! ... j_N!)(j_1^{'}! j_2^{'}! ... j_N^{'}!)}}

    Args:
        initial_fock_state (qutip.Qobj | jnp.ndarray): The initial Fock state.
        final_fock_state (qutip.Qobj | jnp.ndarray): The final Fock state.
        unitary_matrix (jnp.ndarray): The unitary matrix that represents the quantum state circuit.

    Returns:
        float: The transition probability amplitude between the initial and final Fock states.
    """
    columns_indices = fock_state_nonzero_indexes(initial_fock_state)
    rows_indices = fock_state_nonzero_indexes(final_fock_state)

    initial_fock_state_photon_number_factorial = fock_state_to_photon_number_factorial(
        initial_fock_state
    )
    final_fock_state_photon_number_factorial = fock_state_to_photon_number_factorial(
        final_fock_state
    )

    subunitary_selection = subunitary_selection_on_index(
        unitary_matrix=unitary_matrix,
        rows_index=rows_indices,
        columns_index=columns_indices,
    )

    transition_probability_amplitude = unitary_permanent(subunitary_selection)[0] / (
        jnp.sqrt(
            initial_fock_state_photon_number_factorial
            * final_fock_state_photon_number_factorial
        )
    )
    return transition_probability_amplitude
