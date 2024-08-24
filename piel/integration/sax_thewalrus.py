from typing import Any
from ..tools.sax import sax_to_s_parameters_standard_matrix
from ..tools.thewalrus import unitary_permanent


__all__ = ["sax_circuit_permanent", "unitary_permanent"]


def sax_circuit_permanent(
    sax_input: Any,  # sax.SType,
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
