:py:mod:`piel.openlane.utils`
=============================

.. py:module:: piel.openlane.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.utils.find_design_run
   piel.openlane.utils.configure_parametric_designs
   piel.openlane.utils.create_parametric_designs



.. py:function:: find_design_run(design_directory: str | pathlib.Path, run_name: str | None = None) -> str

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.


.. py:function:: configure_parametric_designs(parameter_sweep_dictionary: dict, source_design_directory: str | pathlib.Path) -> list

   For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str | pathlib.Path

   :returns: List of configurations to sweep.
   :rtype: configuration_sweep(list)


.. py:function:: create_parametric_designs(parameter_sweep_dictionary: dict, source_design_directory: str | pathlib.Path, target_directory: str | pathlib.Path) -> None

   Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str
   :param target_directory: Target directory.
   :type target_directory: str

   :returns: None
