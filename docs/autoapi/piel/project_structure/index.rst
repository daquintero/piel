:py:mod:`piel.project_structure`
================================

.. py:module:: piel.project_structure

.. autoapi-nested-parse::

   This file allows us to automate several aspects of creating a fully compatible project structure.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.project_structure.create_setup_py
   piel.project_structure.create_empty_piel_project
   piel.project_structure.get_module_folder_type_location
   piel.project_structure.pip_install_local_module
   piel.project_structure.read_configuration



.. py:function:: create_setup_py(design_directory: piel.types.PathTypes, project_name: Optional[str] = None, from_config_json: bool = True) -> None

   This function creates a setup.py file from the config.json file found in the design directory.

   :param design_directory: Design directory PATH or module name.
   :type design_directory: PathTypes

   :returns: None


.. py:function:: create_empty_piel_project(project_name: str, parent_directory: piel.types.PathTypes) -> None

   This function creates an empty piel-structure project in the target directory. Structuring your files in this way
   enables the co-design and use of the tools supported by piel whilst maintaining the design flow ordered,
   clean and extensible. You can read more about it in the documentation TODO add link.

   TODO just make this a cookiecutter. TO BE DEPRECATED whenever I get round to that.

   :param project_name: Name of the project.
   :type project_name: str
   :param parent_directory: Parent directory of the project.
   :type parent_directory: PathTypes

   :returns: None


.. py:function:: get_module_folder_type_location(module: types.ModuleType, folder_type: Literal[digital_source, digital_testbench])

   This is an easy helper function that saves a particular file in the corresponding location of a `piel` project structure.

   TODO DOCS


.. py:function:: pip_install_local_module(module_path: piel.types.PathTypes)

   This function installs a local module in editable mode.

   :param module_path: Path to the module to be installed.
   :type module_path: PathTypes

   :returns: None


.. py:function:: read_configuration(design_directory: piel.types.PathTypes) -> dict

   This function reads the configuration file found in the design directory.

   :param design_directory: Design directory PATH.
   :type design_directory: PathTypes

   :returns: Configuration dictionary.
   :rtype: config_dictionary(dict)
