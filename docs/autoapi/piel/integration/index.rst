:py:mod:`piel.integration`
==========================

.. py:module:: piel.integration


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   openlane_gdsfactory_core/index.rst
   sax_cocotb/index.rst
   sax_qutip/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.create_gdsfactory_component_from_openlane
   piel.integration.sax_to_ideal_qutip_unitary
   piel.integration.standard_s_parameters_to_ideal_qutip_unitary



.. py:function:: create_gdsfactory_component_from_openlane(design_name_v1: str | None = None, design_directory: piel.config.piel_path_types | None = None, run_name: str | None = None, v1: bool = True) -> gdsfactory.Component

   This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

   It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH.
   :type design_directory: piel_path_types
   :param run_name: Name of the run to extract the GDS from. If None, it will look at the latest run.
   :type run_name: str
   :param v1: If True, it will import the design from the OpenLane v1 configuration.
   :type v1: bool

   :returns: GDSFactory component.
   :rtype: component(gf.Component)


.. py:function:: sax_to_ideal_qutip_unitary(sax_input: sax.SType)

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
       s_gate_matrix = np.array([[1.,   0],
                                [0., 1.j]])
       s_gate = qutip.Qobj(mat, dims=[[2], [2]])

   In mathematical notation, this S-gate would be written as:

   ..math::

       S = \begin{bmatrix}
           1 & 0 \
           0 & i \
       \end{bmatrix}

   :param sax_input: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type sax_input: sax.SType

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)


.. py:function:: standard_s_parameters_to_ideal_qutip_unitary(s_parameters_standard_matrix: piel.config.nso.ndarray)

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
       s_gate_matrix = np.array([[1.,   0],
                                [0., 1.j]])
       s_gate = qutip.Qobj(mat, dims=[[2], [2]])

   In mathematical notation, this S-gate would be written as:

   ..math::

       S = \begin{bmatrix}
           1 & 0 \
           0 & i \
       \end{bmatrix}

   :param s_parameters_standard_matrix: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type s_parameters_standard_matrix: nso.ndarray

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)
