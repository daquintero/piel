:py:mod:`piel.integration.sax_qutip`
====================================

.. py:module:: piel.integration.sax_qutip


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.sax_qutip.sax_to_ideal_qutip_unitary
   piel.integration.sax_qutip.verify_sax_model_is_unitary



.. py:function:: sax_to_ideal_qutip_unitary(sax_input: sax.SType, input_ports_order: tuple | None = None)

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

       S = \begin{bmatrix}
           1 & 0 \\
           0 & i \\
       \end{bmatrix}

   :param sax_input: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type sax_input: sax.SType
   :param input_ports_order: The order of the input ports. If None, the default order is used.
   :type input_ports_order: tuple | None

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)


.. py:function:: verify_sax_model_is_unitary(model: sax.SType, input_ports_order: tuple | None = None) -> bool

   Verify that the model is unitary.

   :param model: The model to verify.
   :type model: dict
   :param input_ports_order: The order of the input ports. If None, the default order is used.
   :type input_ports_order: tuple | None

   :returns: True if the model is unitary, False otherwise.
   :rtype: bool
