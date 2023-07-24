:py:mod:`piel.tools.thewalrus.operations`
=========================================

.. py:module:: piel.tools.thewalrus.operations


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.thewalrus.operations.unitary_permanent



.. py:function:: unitary_permanent(unitary_matrix: jax.numpy.ndarray) -> tuple

   The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

   ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

   Note that this function needs to be as optimised as possible, so we need to minimise our computational complexity of our operation.

   # TODO implement validation
   # TODO maybe implement subroutine if computation is taking forever.
   # TODO why two outputs? Understand this properly later.

   :param unitary_permanent: The unitary matrix.
   :type unitary_permanent: np.ndarray

   :returns: The circuit permanent and the time it took to compute it.
   :rtype: tuple
