import qutip  # NOQA : F401
import sax
from ..config import nso
from piel.tools.sax.utils import sax_to_s_parameters_standard_matrix

__all__ = ["sax_to_ideal_qutip_unitary", "standard_s_parameters_to_ideal_qutip_unitary"]


def standard_s_parameters_to_ideal_qutip_unitary(
    s_parameters_standard_matrix: nso.ndarray,
):
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
    Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
    already in described in piel/piel/sax/utils.py.

    From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly. https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

    For example, a ``qutip`` representation of an s-gate gate would be:

    ..code-block:: python

        import numpy as np
        import qutip

        # S-Gate
        s_gate_matrix = np.array([[1.,   0], [0., 1.j]])
        s_gate = qutip.Qobj(mat, dims=[[2], [2]])

    In mathematical notation, this S-gate would be written as:

    ..math::

        S = \\begin{bmatrix}
            1 & 0 \\
            0 & i \\
        \\end{bmatrix}

    Args:
        s_parameters_standard_matrix (nso.ndarray): A dictionary of S-parameters in the form of a SDict from `sax`.

    Returns:
        qobj_unitary (qutip.Qobj): A QuTip QObj representation of the S-parameters in a unitary matrix.

    """
    # TODO make a function any SAX input.
    qobj_unitary = qutip.Qobj(s_parameters_standard_matrix)
    return qobj_unitary


def sax_to_ideal_qutip_unitary(sax_input: sax.SType):
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
    Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
    already in described in piel/piel/sax/utils.py.

    From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly.
    https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

    For example, a ``qutip`` representation of an s-gate gate would be:

    ..code-block:: python

        import numpy as np
        import qutip
        # S-Gate
        s_gate_matrix = np.array([[1.,   0], [0., 1.j]])
        s_gate = qutip.Qobj(mat, dims=[[2], [2]])

    In mathematical notation, this S-gate would be written as:

    ..math::

        S = \\begin{bmatrix}
            1 & 0 \\
            0 & i \\
        \\end{bmatrix}

    Args:
        sax_input (sax.SType): A dictionary of S-parameters in the form of a SDict from `sax`.

    Returns:
        qobj_unitary (qutip.Qobj): A QuTip QObj representation of the S-parameters in a unitary matrix.
    """
    # TODO make a function any SAX input.
    (
        s_parameters_standard_matrix,
        input_ports_index_tuple_order,
    ) = sax_to_s_parameters_standard_matrix(sax_input)
    qobj_unitary = standard_s_parameters_to_ideal_qutip_unitary(
        s_parameters_standard_matrix
    )
    return qobj_unitary
