``qutip`` - ``sax``
===================

One interesting relationship to explore is how a particular extracted
unitary from a photonic network affect the quantum states (such as
photons) it operates upon. This is the main motivation for integrating
``qutip`` and ``sax``.

We provide an example of this integration here:


Conversion Assumptions
----------------------

There are a list of assumptions that the user must be aware when converting in between different modelling regimes.

Quantum Information Basics
--------------------------

Properties of unitary matrices:

.. list-table:: Unitary Properties
   :header-rows: 1

   * - Unitary Property
     - Notation
   * - Unitary
     - :math:`U^\dagger U = I`
   * - Normal
     - :math:`U U^\dagger = U^\dagger U = I`
   * - Invertible
     - :math:`U^\dagger = U^{-1}`
   * - Unit Eigenvalues
     - :math:`\| \lambda \| ^2 = 1`
   * - Length Preserving
     - :math:`\langle U \psi \rangle = \langle \psi \rangle`

Properties of Hamiltonian matrices:

.. list-table:: Hamiltonian Properties
   :align: center

   * - Unitary Property
     - Notation
   * - Hermitian
     - :math:`H^\dagger = H`
   * - Normal
     - :math:`H H^\dagger = H^\dagger H`

.. TODO check this.
.. \text{Real Eigenvalues} & \lambda \in \mathbb{R} \\
.. \text{Orthonormal Eigenvectors} & \langle \psi_i | \psi_j \rangle = \delta_{ij} \\
.. \text{Unitary Eigenbasis} & U^\dagger = U^{-1} \\
