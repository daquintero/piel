:py:mod:`piel.openlane`
=======================

.. py:module:: piel.openlane


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   v1_parse/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   migrate/index.rst
   utils/index.rst
   v1/index.rst
   v2/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.get_design_directory_from_root_openlane_v1
   piel.openlane.return_path
   piel.openlane.get_design_from_openlane_migration
   piel.openlane.find_design_run
   piel.openlane.check_config_json_exists_openlane_v1
   piel.openlane.check_design_exists_openlane_v1
   piel.openlane.configure_and_run_design_openlane_v1
   piel.openlane.configure_parametric_designs_openlane_v1
   piel.openlane.configure_flow_script_openlane_v1
   piel.openlane.create_parametric_designs_openlane_v1
   piel.openlane.get_design_directory_from_root_openlane_v1
   piel.openlane.get_latest_version_root_openlane_v1
   piel.openlane.read_configuration_openlane_v1
   piel.openlane.write_configuration_openlane_v1
   piel.openlane.run_openlane_flow



.. py:function:: get_design_directory_from_root_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> pathlib.Path

   Gets the design directory from the root directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Design directory.
   :rtype: design_directory(pathlib.Path)


.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: get_design_from_openlane_migration(v1: bool = True, design_name_v1: str | None = None, design_directory: str | pathlib.Path | None = None, root_directory_v1: str | pathlib.Path | None = None) -> (str, pathlib.Path)

   This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

   This function checks if the inputs are to be treated as v1 inputs. If so, and a `design_name` is provided then it will set the `design_directory` to the corresponding `design_name` directory in the corresponding `root_directory_v1 / designs`. If no `root_directory` is provided then it returns `$OPENLANE_ROOT/"<latest>"/. If a `design_directory` is provided then this will always take precedence even with a `v1` flag.

   :param v1: If True, it will migrate from v1 to v2.
   :type v1: bool
   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH. Optional path for v2-based designs.
   :type design_directory: str
   :param root_directory_v1: Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"
   :type root_directory_v1: str

   :returns: None


.. py:function:: find_design_run(design_directory: str | pathlib.Path, run_name: str | None = None) -> pathlib.Path

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.


.. py:function:: check_config_json_exists_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> bool

   Checks if a design has a `config.json` file.

   :param design_name: Name of the design.
   :type design_name: str

   :returns: True if `config.json` exists.
   :rtype: config_json_exists(bool)


.. py:function:: check_design_exists_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> bool

   Checks if a design exists in the OpenLane v1 design folder.

   Lists all designs inside the Openlane V1 design root.

   :param design_name: Name of the design.
   :type design_name: str

   :returns: True if design exists.
   :rtype: design_exists(bool)


.. py:function:: configure_and_run_design_openlane_v1(design_name: str, configuration: dict | None = None, root_directory: str | pathlib.Path | None = None) -> None

   Configures and runs an OpenLane v1 design.

   This function does the following:
   1. Check that the design_directory provided is under $OPENLANE_ROOT/<latestversion>/designs
   2. Check if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`.
   3. Create a script directory, a script is written and permissions are provided for it to be executable.
   4. Permit and execute the `openlane_flow.sh` script in the `scripts` directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param configuration: Configuration dictionary.
   :type configuration: dict | None
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: None


.. py:function:: configure_parametric_designs_openlane_v1(design_name: str, parameter_sweep_dictionary: dict, add_id: bool = True) -> list

   For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

   :param add_id: Add an ID to the design name. Defaults to True.
   :type add_id: bool
   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str | pathlib.Path

   :returns: List of configurations to sweep.
   :rtype: configuration_sweep(list)


.. py:function:: configure_flow_script_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> None

   Configures the OpenLane v1 flow script after checking that the design directory exists.

   :param design_directory: Design directory. Defaults to latest OpenLane root.
   :type design_directory: str | pathlib.Path | None

   :returns: None


.. py:function:: create_parametric_designs_openlane_v1(design_name: str, parameter_sweep_dictionary: dict, target_directory: str | pathlib.Path | None = None) -> None

   Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

   :param design_name: Name of the design.
   :type design_name: str
   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param target_directory: Optional target directory.
   :type target_directory: str | pathlib.Path | None

   :returns: None


.. py:function:: get_design_directory_from_root_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> pathlib.Path

   Gets the design directory from the root directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Design directory.
   :rtype: design_directory(pathlib.Path)


.. py:function:: get_latest_version_root_openlane_v1() -> pathlib.Path

   Gets the latest version root of OpenLane v1.


.. py:function:: read_configuration_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> dict

   Reads a `config.json` from a design directory.

   :param design_name: Design name.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Configuration dictionary.
   :rtype: configuration(dict)


.. py:function:: write_configuration_openlane_v1(configuration: dict, design_directory: str | pathlib.Path) -> None

   Writes a `config.json` onto a `design_directory`

   :param configuration: OpenLane configuration dictionary.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


.. py:function:: run_openlane_flow(configuration: dict | None = test_spm_open_lane_configuration, design_directory: str = '/foss/designs/spm') -> None

   Runs the OpenLane flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None
