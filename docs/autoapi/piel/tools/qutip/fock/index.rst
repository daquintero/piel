:py:mod:`piel.tools.qutip.fock`
===============================

.. py:module:: piel.tools.qutip.fock


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.fock.fock_state_to_photon_number_factorial
   piel.tools.qutip.fock.fock_state_nonzero_indexes



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



.. py:function:: fock_state_nonzero_indexes(fock_state: qutip.Qobj)

   This function returns the indexes of the nonzero elements of a Fock state.

   :param fock_state: A QuTip QObj representation of the Fock state.
   :type fock_state: qutip.Qobj

   :returns: The indexes of the nonzero elements of the Fock state.
   :rtype: tuple
