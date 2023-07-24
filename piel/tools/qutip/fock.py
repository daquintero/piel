import math
import qutip

__all__ = ["fock_state_nonzero_indexes", "fock_state_to_photon_number_factorial"]


def fock_state_to_photon_number_factorial(fock_state: qutip.Qobj):
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
    photon_number_factorial = 1
    for photon_number in fock_state:
        photon_number_amount = photon_number[0].real
        photon_number_factorial *= math.factorial(int(photon_number_amount[0]))
    return photon_number_factorial


def fock_state_nonzero_indexes(fock_state: qutip.Qobj):
    """
    This function returns the indexes of the nonzero elements of a Fock state.

    Args:
        fock_state (qutip.Qobj): A QuTip QObj representation of the Fock state.

    Returns:
        tuple: The indexes of the nonzero elements of the Fock state.
    """
    # TODO implement checks of Fock state validity
    nonzero_indexes = []
    for index, photon_number in enumerate(fock_state):
        photon_number_amount = photon_number[0].real
        if photon_number_amount[0] != 0:
            nonzero_indexes.append(index)
    return tuple(nonzero_indexes)
