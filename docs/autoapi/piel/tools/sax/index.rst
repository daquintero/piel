:py:mod:`piel.tools.sax`
========================

.. py:module:: piel.tools.sax


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   netlist/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.sax.address_value_dictionary_to_function_parameter_dictionary
   piel.tools.sax.compose_recursive_instance_location
   piel.tools.sax.get_component_instances
   piel.tools.sax.get_netlist_instances_by_prefix
   piel.tools.sax.get_matched_model_recursive_netlist_instances
   piel.tools.sax.get_sdense_ports_index
   piel.tools.sax.sax_to_s_parameters_standard_matrix



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.sax.snet


.. py:function:: address_value_dictionary_to_function_parameter_dictionary(address_value_dictionary: dict, parameter_key: str)

   This function converts an address of an instance with particular parameter values in the form:

       {('component_lattice_gener_fb8c4da8', 'mzi_1', 'sxt'): 0,
       ('component_lattice_gener_fb8c4da8', 'mzi_5', 'sxt'): 0}

   to

       {'mzi_1': {'sxt': {parameter_key: 0}},
       ('mzi_5', {'sxt': {parameter_key: 0}}}




.. py:function:: compose_recursive_instance_location(recursive_netlist: dict, top_level_instance_name: str, required_models: list, target_component_prefix: str, models: dict)

      This function returns the recursive location of any matching ``target_component_prefix`` instances within the ``recursive_netlist``. A function that returns the mapping of the ``matched_component`` in the corresponding netlist at any particular level of recursion. This function iterates over a particular level of recursion of a netlist. It returns a list of the missing required components, and updates a dictionary of models that contains a particular matching component. It returns the corresponding list of instances of a particular component at that level of recursion, so that it can be appended upon in order to construct the location of the corresponding matching elements.

      If ``required_models`` is an empty list, it means no recursion is required and the function is complete. If a ``required_model_i`` in ``required_models`` matches ``target_component_prefix``, then no more recursion is required down the component function.

      The ``recursive_netlist`` should contain all the missing composed models that are not provided in the main models dictionary. If not, then we need to require the user to input the missing model that cannot be extracted from the composed netlist.
   We know when a model is composed, and when it is already provided at every level of recursion based on the ``models`` dictionary that gets updated at each level of recursion with the corresponding models of that level, and the ``required_models`` down itself.

      However, a main question appears on how to do the recursion. There needs to be a flag that determines that the recursion is complete. However, this is only valid for every particular component in the ``required_models`` list. Every component might have missing component. This means that this recursion begins component by component, updating the ``required_models`` list until all of them have been composed from the recursion or it is determined that is it missing fully.

      It would be ideal to access the particular component that needs to be implemented.

      Returns a tuple of ``model_composition_mapping, instance_composition_mapping, target_component_mapping`` in the form of

          ({'mzi_214beef3': ['straight_heater_metal_s_ad3c1693']},
           {'mzi_214beef3': ['mzi_1', 'mzi_5'],
            'mzi_d46c281f': ['mzi_2', 'mzi_3', 'mzi_4']})


.. py:function:: get_component_instances(recursive_netlist: dict, top_level_prefix: str, component_name_prefix: str)

   Returns a dictionary of all instances of a given component in a recursive netlist.

   :param recursive_netlist: The recursive netlist to search.
   :param top_level_prefix: The prefix of the top level instance.
   :param component_name_prefix: The name of the component to search for.

   :returns: A dictionary of all instances of the given component.


.. py:function:: get_netlist_instances_by_prefix(recursive_netlist: dict, instance_prefix: str) -> str

   Returns a list of all instances with a given prefix in a recursive netlist.

   :param recursive_netlist: The recursive netlist to search.
   :param instance_prefix: The prefix to search for.

   :returns: A list of all instances with the given prefix.


.. py:function:: get_matched_model_recursive_netlist_instances(recursive_netlist: dict, top_level_instance_prefix: str, target_component_prefix: str, models: Optional[dict] = None) -> list[tuple]

   This function returns an active component list with a tuple mapping of the location of the active component within the recursive netlist and corresponding model. It will recursively look within a netlist to locate what models use a particular component model. At each stage of recursion, it will compose a list of the elements that implement this matching model in order to relate the model to the instance, and hence the netlist address of the component that needs to be updated in order to functionally implement the model.

   It takes in as a set of parameters the recursive_netlist generated by a ``gdsfactory`` netlist implementation.

   Returns a list of tuples, that correspond to the phases applied with the corresponding component paths at multiple levels of recursion.
   eg. [("component_lattice_gener_fb8c4da8", "mzi_1", "sxt"), ("component_lattice_gener_fb8c4da8", "mzi_5", "sxt")] and these are our keys to our sax circuit decomposition.


.. py:function:: get_sdense_ports_index(input_ports_order: tuple, all_ports_index: dict) -> dict

   This function returns the ports index of the sax dense S-parameter matrix.

   Given that the order of the iteration is provided by the user, the dictionary keys will also be ordered
   accordingly when iterating over them. This requires the user to provide a set of ordered.

   TODO verify reasonable iteration order.

   .. code-block:: python

       # The input_ports_order can be a tuple of tuples that contain the index and port name. Eg.
       input_ports_order = ((0, "in_o_0"), (5, "in_o_1"), (6, "in_o_2"), (7, "in_o_3"))
       # The all_ports_index is a dictionary of the ports index. Eg.
       all_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }
       # Output
       {"in_o_0": 0, "in_o_1": 5, "in_o_2": 6, "in_o_3": 7}

   :param input_ports_order: The ports order tuple. Can be a tuple of tuples that contain the index and port name.
   :type input_ports_order: tuple
   :param all_ports_index: The ports index dictionary.
   :type all_ports_index: dict

   :returns: The ordered input ports index tuple.
   :rtype: tuple


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


.. py:data:: snet
