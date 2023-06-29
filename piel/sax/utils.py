"""
This file provides a set of utilities that allow much easier integration between `sax` and the relevant tools that we use.
"""
import sax
from typing import Optional  # NOQA : F401
from ..config import nso

__all__ = ["get_sdense_ports_index", "sax_s_parameters_to_matrix", "snet"]


def get_sdense_ports_index(input_ports_order: tuple, all_ports_index: dict) -> dict:
    """
    This function returns the ports index of the sax dense S-parameter matrix.

    Given that the order of the iteration is provided by the user, the dictionary keys will also be ordered
    accordingly when iterating over them.

    TODO verify reasonable iteration order.

    Args:
        input_ports_order (tuple): The ports order tuple.
        all_ports_index (dict): The ports index dictionary.

    Returns:
        tuple: The ordered input ports index tuple.
    """
    # TODO look into jnp.at method https://github.com/flaport/sax/blob/a87c3bf8c792dc227779e5d010627897f4cd8278/sax/typing_.py#L355
    input_ports_index = {key: all_ports_index[key] for key in input_ports_order}
    return input_ports_index


def sax_s_parameters_to_matrix(
    sdict=sax.SDict,
) -> nso.ndarray:
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    A ``sax`` S-parameter dictionary is provided as a dictionary of tuples with (port0, port1) as the key. This
    determines the direction of the scattering relationship. It means that the number of terms in an S-parameter
    matrix is the number of ports squared.

    We need to define the input ports, and the output ports for this structure to be generated.

    A S-Parameter matrix in the form is returned:

    ..math::

        S = \\begin{bmatrix}
            S_{11} & S_{12} & S_{13} & S_{14} \\
            S_{21} & S_{22} & S_{23} & S_{24} \\
            S_{31} & S_{32} & S_{33} & S_{34} \\
            S_{41} & S_{42} & S_{43} & S_{44} \\
        \\end{bmatrix}

    """
    dense_s_parameter_matrix, dense_s_parameter_index = sax.sdense(sdict)
    # Now we get the indexes of the input ports that we care about to restructure the dense matrix with the columns we care about.

    get_sdense_ports_index(all_ports_index=dense_s_parameter_index)
    pass


snet = sax_s_parameters_to_matrix
