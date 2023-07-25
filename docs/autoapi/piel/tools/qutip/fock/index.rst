:py:mod:`piel.tools.qutip.fock`
===============================

.. py:module:: piel.tools.qutip.fock


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.qutip.fock.all_fock_states_from_photon_number
   piel.tools.qutip.fock.fock_state_to_photon_number_factorial
   piel.tools.qutip.fock.fock_state_nonzero_indexes
   piel.tools.qutip.fock.fock_states_at_mode_index



.. py:function:: all_fock_states_from_photon_number(mode_amount: int, photon_amount: int = 1) -> list[qutip.Qobj]

   For a specific amount of modes, we can generate all the possible Fock states for whatever amount of input photons we desire. This returns a list of all corresponding Fock states.

   :param mode_amount: The amount of modes in the system.
   :type mode_amount: int
   :param photon_amount: The amount of photons in the system. Defaults to 1.
   :type photon_amount: int, optional

   :returns: A list of all the Fock states.
   :rtype: list


.. py:function:: fock_state_to_photon_number_factorial(fock_state: qutip.Qobj) -> float

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



.. py:function:: fock_state_nonzero_indexes(fock_state: qutip.Qobj) -> tuple[int]

   This function returns the indexes of the nonzero elements of a Fock state.

   :param fock_state: A QuTip QObj representation of the Fock state.
   :type fock_state: qutip.Qobj

   :returns: The indexes of the nonzero elements of the Fock state.
   :rtype: tuple


.. py:function:: fock_states_at_mode_index(mode_amount: int, target_mode_index: int, maximum_photon_amount: Optional[int] = 1) -> list[qutip.Qobj]

   This function returns a list of valid Fock states that fulfill a condition of having a maximum photon number at a specific mode index.

   :param mode_amount: The amount of modes in the system.
   :type mode_amount: int
   :param target_mode_index: The mode index to check the photon number at.
   :type target_mode_index: int
   :param maximum_photon_amount: The amount of photons in the system. Defaults to 1.
   :type maximum_photon_amount: int, optional

   :returns: A list of all the Fock states.
   :rtype: list
