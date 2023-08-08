import qutip  # NOQA : F401
import sax
from ..tools.qutip.unitary import matrix_to_qutip_qobj
from piel.tools.sax.utils import sax_to_s_parameters_standard_matrix

__all__ = [
    "sax_to_ideal_qutip_unitary",
    "verify_sax_model_is_unitary",
]


def sax_to_ideal_qutip_unitary(
    sax_input: sax.SType, input_ports_order: tuple | None = None
):
    """
    This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
    dimensions of the matrix can be observed.

    I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
    Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
    already in described in piel/piel/sax/utils.py.

    From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly.
    https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

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
        sax_input (sax.SType): A dictionary of S-parameters in the form of a SDict from `sax`.
        input_ports_order (tuple | None): The order of the input ports. If None, the default order is used.

    Returns:
        qobj_unitary (qutip.Qobj): A QuTip QObj representation of the S-parameters in a unitary matrix.
    """
    # TODO make a function any SAX input.
    (
        s_parameters_standard_matrix,
        input_ports_index_tuple_order,
    ) = sax_to_s_parameters_standard_matrix(
        sax_input=sax_input, input_ports_order=input_ports_order
    )
    qobj_unitary = matrix_to_qutip_qobj(s_parameters_standard_matrix)
    return qobj_unitary


def verify_sax_model_is_unitary(
    model: sax.SType, input_ports_order: tuple | None = None
) -> bool:
    """
    Verify that the model is unitary.

    Args:
        model (dict): The model to verify.
        input_ports_order (tuple | None): The order of the input ports. If None, the default order is used.

    Returns:
        bool: True if the model is unitary, False otherwise.
    """
    qobj = sax_to_ideal_qutip_unitary(model, input_ports_order=input_ports_order)
    return qobj.check_isunitary()
