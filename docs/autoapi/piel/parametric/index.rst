:py:mod:`piel.parametric`
=========================

.. py:module:: piel.parametric


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.parametric.single_parameter_sweep
   piel.parametric.multi_parameter_sweep



.. py:function:: single_parameter_sweep(base_design_configuration: dict, parameter_name: str, parameter_sweep_values: list)

   This function takes a base_design_configuration dictionary and sweeps a single parameter over a list of values. It returns a list of dictionaries that correspond to the parameter sweep.

   :param base_design_configuration: Base design configuration dictionary.
   :type base_design_configuration: dict
   :param parameter_name: Name of parameter to sweep.
   :type parameter_name: str
   :param parameter_sweep_values: List of values to sweep.
   :type parameter_sweep_values: list

   :returns: List of dictionaries that correspond to the parameter sweep.
   :rtype: parameter_sweep_design_dictionary_array(list)


.. py:function:: multi_parameter_sweep(base_design_configuration: dict, parameter_sweep_dictionary: dict) -> list

   This multiparameter sweep is pretty cool, as it will generate designer list of dictionaries that comprise of all the possible combinations of your parameter sweeps. For example, if you are sweeping `parameter_1 = np.arange(0, 2) = array([0, 1])`, and `parameter_2 = np.arange(2, 4) = array([2, 3])`, then this function will generate list of dictionaries based on the default_design dictionary, but that will comprise of all the potential parameter combinations within this list.

   For the example above, there arould be 4 combinations [(0, 2), (0, 3), (1, 2), (1, 3)].

   If you were instead sweeping for `parameter_1 = np.arange(0, 5)` and `parameter_2 = np.arange(0, 5)`, the dictionary generated would correspond to these parameter combinations of::
       [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)].

   Make sure to use the parameter_names from default_design when writing up the parameter_sweep dictionary key name.

   Example project_structure formats::

       example_parameter_sweep_dictionary = {
           "parameter_1": np.arange(1, -40, 1),
           "parameter_2": np.arange(1, -40, 1),
       }

       example_base_design_configuration = {
           "parameter_1": 10.0,
           "parameter_2": 40.0,
           "parameter_3": 0,
       }

   :param base_design_configuration: Dictionary of the default design configuration.
   :type base_design_configuration: dict
   :param parameter_sweep_dictionary: Dictionary of the parameter sweep. The keys should be the same as the keys in the base_design_configuration dictionary.
   :type parameter_sweep_dictionary: dict

   :returns: List of dictionaries that comprise of all the possible combinations of your parameter sweeps.
   :rtype: parameter_sweep_design_dictionary_array(list)
