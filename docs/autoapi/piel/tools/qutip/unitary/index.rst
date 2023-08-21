:py:mod:`piel.tools.qutip.unitary`
==================================

.. py:module:: piel.tools.qutip.unitary


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.unitary.subunitary_selection_on_index
   piel.tools.qutip.unitary.subunitary_selection_on_range
   piel.tools.qutip.unitary.verify_matrix_is_unitary



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.unitary.standard_s_parameters_to_qutip_qobj


.. py:function:: subunitary_selection_on_index(unitary_matrix: jax.numpy.ndarray, rows_index: jax.numpy.ndarray | tuple, columns_index: jax.numpy.ndarray | tuple)

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.


.. py:function:: subunitary_selection_on_range(unitary_matrix: jax.numpy.ndarray, stop_index: tuple, start_index: Optional[tuple] = (0, 0))

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.


.. py:function:: verify_matrix_is_unitary(matrix: jax.numpy.ndarray) -> bool

   Verify that the matrix is unitary.

   :param matrix: The matrix to verify.
   :type matrix: jnp.ndarray

   :returns: True if the matrix is unitary, False otherwise.
   :rtype: bool


.. py:data:: standard_s_parameters_to_qutip_qobj


