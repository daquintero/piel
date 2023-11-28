:py:mod:`piel.tools.sax.netlist`
================================

.. py:module:: piel.tools.sax.netlist

.. autoapi-nested-parse::

   This module aims to extend sax from standard netlist operations to include more complex operations that enable connectivity.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.sax.netlist.address_value_dictionary_to_function_parameter_dictionary
   piel.tools.sax.netlist.compose_recursive_instance_location
   piel.tools.sax.netlist.get_component_instances
   piel.tools.sax.netlist.get_netlist_instances_by_prefix
   piel.tools.sax.netlist.get_matched_model_recursive_netlist_instances



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


