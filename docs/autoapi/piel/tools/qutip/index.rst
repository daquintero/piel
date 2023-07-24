:py:mod:`piel.tools.qutip`
==========================

.. py:module:: piel.tools.qutip


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   fock/index.rst
   unitary/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.fock_state_nonzero_indexes
   piel.tools.qutip.fock_state_to_photon_number_factorial
   piel.tools.qutip.verify_matrix_is_unitary
   piel.tools.qutip.subunitary_selection_on_range
   piel.tools.qutip.subunitary_selection_on_index



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.standard_s_parameters_to_qutip_qobj


.. py:function:: fock_state_nonzero_indexes(fock_state: qutip.Qobj)

   This function returns the indexes of the nonzero elements of a Fock state.

   :param fock_state: A QuTip QObj representation of the Fock state.
   :type fock_state: qutip.Qobj

   :returns: The indexes of the nonzero elements of the Fock state.
   :rtype: tuple


.. py:function:: fock_state_to_photon_number_factorial(fock_state: qutip.Qobj)

       This function converts a Fock state defined as:

       .. math::

   ewcommand{\ket}[1]{\left|{#1}
   ight
   angle}
           \ket{f_1} = \ket{j_1, j_2, ... j_N}$

       and returns:

       .. math::

           j_1^{'}! j_2^{'}! ... j_N^{'}!

       Args:
           fock_state (qutip.Qobj): A QuTip QObj representation of the Fock state.

       Returns:
           float: The photon number factorial of the Fock state.



.. py:data:: standard_s_parameters_to_qutip_qobj



.. py:function:: verify_matrix_is_unitary(matrix: jax.numpy.ndarray) -> bool

   Verify that the matrix is unitary.

   :param matrix: The matrix to verify.
   :type matrix: jnp.ndarray

   :returns: True if the matrix is unitary, False otherwise.
   :rtype: bool


.. py:function:: subunitary_selection_on_range(unitary_matrix: jax.numpy.ndarray, stop_index: tuple, start_index: Optional[tuple] = (0, 0))

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.


.. py:function:: subunitary_selection_on_index(unitary_matrix: jax.numpy.ndarray, rows_index: jax.numpy.ndarray | tuple, columns_index: jax.numpy.ndarray | tuple)

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.
