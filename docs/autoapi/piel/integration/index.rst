:py:mod:`piel.integration`
==========================

.. py:module:: piel.integration


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   gdsfactory_hdl21/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   cocotb_sax/index.rst
   gdsfactory_openlane/index.rst
   sax_qutip/index.rst
   sax_thewalrus/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.create_gdsfactory_component_from_openlane
   piel.integration.gdsfactory_netlist_to_spice_netlist
   piel.integration.construct_hdl21_module
   piel.integration.convert_connections_to_tuples
   piel.integration.gdsfactory_netlist_with_hdl21_generators
   piel.integration.sax_to_s_parameters_standard_matrix
   piel.integration.unitary_permanent
   piel.integration.sax_circuit_permanent
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


.. py:function:: gdsfactory_netlist_to_spice_netlist(gdsfactory_netlist: dict, generators: dict, **kwargs) -> hdl21.Module

   This function converts a GDSFactory electrical netlist into a standard SPICE netlist. It follows the same
   principle as the `sax` circuit composition.

   Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
   set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
   the instance model we provides.

   We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own
   composition circuit. Write the SPICE component based on the model into a total circuit representation in string
   from the reshaped gdsfactory dictionary into our own structure.

   :param gdsfactory_netlist: GDSFactory netlist
   :param generators: Dictionary of Generators

   :returns: hdl21 module or raw SPICE string


.. py:function:: construct_hdl21_module(spice_netlist: dict, **kwargs) -> hdl21.Module

   This function converts a gdsfactory-spice converted netlist using the component models into a SPICE circuit.

   Part of the complexity of this function is the multiport nature of some components and models, and assigning the
   parameters accordingly into the SPICE function. This is because not every SPICE component will be bi-port,
   and many will have multi-ports and parameters accordingly. Each model can implement the composition into a
   SPICE circuit, but they depend on a set of parameters that must be set from the instance. Another aspect is
   that we may want to assign the component ID according to the type of component. However, we can also assign the
   ID based on the individual instance in the circuit, which is also a reasonable approximation. However,
   it could be said, that the ideal implementation would be for each component model provided to return the SPICE
   instance including connectivity except for the ID.

   # TODO implement validators


.. py:function:: convert_connections_to_tuples(connections: dict)

   Convert from:

   .. code-block::

       {
       'straight_1,e1': 'taper_1,e2',
       'straight_1,e2': 'taper_2,e2',
       'taper_1,e1': 'via_stack_1,e3',
       'taper_2,e1': 'via_stack_2,e1'
       }

   to:

   .. code-block::

       [(('straight_1', 'e1'), ('taper_1', 'e2')), (('straight_1', 'e2'), ('taper_2', 'e2')), (('taper_1', 'e1'),
       ('via_stack_1', 'e3')), (('taper_2', 'e1'), ('via_stack_2', 'e1'))]


.. py:function:: gdsfactory_netlist_with_hdl21_generators(gdsfactory_netlist: dict, generators=None)

   This function allows us to map the ``hdl21`` models dictionary in a `sax`-like implementation to the ``GDSFactory`` netlist. This allows us to iterate over each instance in the netlist and construct a circuit after this function.]

   Example usage:

   .. code-block::

       >>> import gdsfactory as gf
       >>> from piel.integration.gdsfactory_hdl21.conversion import gdsfactory_netlist_with_hdl21_generators
       >>> from piel.models.physical.electronic import get_default_models
       >>> gdsfactory_netlist_with_hdl21_generators(gdsfactory_netlist=gf.components.mzi2x2_2x2_phase_shifter().get_netlist(exclude_port_types="optical"),generators=get_default_models())

   :param gdsfactory_netlist: The netlist from ``GDSFactory`` to map to the ``hdl21`` models dictionary.
   :param generators: The ``hdl21`` models dictionary to map to the ``GDSFactory`` netlist.

   :returns: The ``GDSFactory`` netlist with the ``hdl21`` models dictionary.


.. py:function:: sax_to_s_parameters_standard_matrix(sax_input: sax.SType, input_ports_order: tuple | None = None) -> tuple

   A ``sax`` S-parameter SDict is provided as a dictionary of tuples with (port0, port1) as the key. This
   determines the direction of the scattering relationship. It means that the number of terms in an S-parameter
   matrix is the number of ports squared.

   In order to generalise, this function returns both the S-parameter matrices and the indexing ports based on the
   amount provided. In terms of computational speed, we definitely would like this function to be algorithmically
   very fast. For now, I will write a simple python implementation and optimise in the future.

   It is possible to see the `sax` SDense notation equivalence here:
   https://flaport.github.io/sax/nbs/08_backends.html

   .. code-block:: python

       import jax.numpy as jnp
       from sax.core import SDense

       # Directional coupler SDense representation
       dc_sdense: SDense = (
           jnp.array([[0, 0, τ, κ], [0, 0, κ, τ], [τ, κ, 0, 0], [κ, τ, 0, 0]]),
           {"in0": 0, "in1": 1, "out0": 2, "out1": 3},
       )


       # Directional coupler SDict representation
       # Taken from https://flaport.github.io/sax/nbs/05_models.html
       def coupler(*, coupling: float = 0.5) -> SDict:
           kappa = coupling**0.5
           tau = (1 - coupling) ** 0.5
           sdict = reciprocal(
               {
                   ("in0", "out0"): tau,
                   ("in0", "out1"): 1j * kappa,
                   ("in1", "out0"): 1j * kappa,
                   ("in1", "out1"): tau,
               }
           )
           return sdict

   If we were to relate the mapping accordingly based on the ports indexes, a S-Parameter matrix in the form of
   :math:`S_{(output,i),(input,i)}` would be:

   .. math::

       S = \begin{bmatrix}
               S_{00} & S_{10} \\
               S_{01} & S_{11} \\
           \end{bmatrix} =
           \begin{bmatrix}
           \tau & j \kappa \\
           j \kappa & \tau \\
           \end{bmatrix}

   Note that the standard S-parameter and hence unitary representation is in the form of:

   .. math::

       S = \begin{bmatrix}
               S_{00} & S_{01} \\
               S_{10} & S_{11} \\
           \end{bmatrix}


   .. math::

       \begin{bmatrix}
           b_{1} \\
           \vdots \\
           b_{n}
       \end{bmatrix}
       =
       \begin{bmatrix}
           S_{11} & \dots & S_{1n} \\
           \vdots & \ddots & \vdots \\
           S_{n1} & \dots & S_{nn}
       \end{bmatrix}
       \begin{bmatrix}
           a_{1} \\
           \vdots \\
           a_{n}
       \end{bmatrix}

   TODO check with Floris, does this mean we need to transpose the matrix?

   :param sax_input: The sax S-parameter dictionary.
   :type sax_input: sax.SType
   :param input_ports_order: The ports order tuple containing the names and order of the input ports.
   :type input_ports_order: tuple

   :returns: The S-parameter matrix and the input ports index tuple in the standard S-parameter notation.
   :rtype: tuple


.. py:function:: unitary_permanent(unitary_matrix: numpy.ndarray) -> tuple

   The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

   ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

   # TODO maybe implement subroutine if computation is taking forever.
   # TODO why two outputs? Understand this properly later.

   :param unitary_permanent: The unitary matrix.
   :type unitary_permanent: np.ndarray

   :returns: The circuit permanent and the time it took to compute it.
   :rtype: tuple


.. py:function:: sax_circuit_permanent(sax_input: sax.SType) -> tuple

   The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

   ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

   # TODO maybe implement subroutine if computation is taking forever.

   :param sax_input: The sax S-parameter dictionary.
   :type sax_input: sax.SType

   :returns: The circuit permanent and the time it took to compute it.
   :rtype: tuple


.. py:function:: sax_to_ideal_qutip_unitary(sax_input: sax.SType)

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

   :param s_parameters_standard_matrix: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type s_parameters_standard_matrix: nso.ndarray

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)
