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

   piel.project_structure.read_configuration
   piel.project_structure.create_setup_py_from_config_json



.. py:function:: read_configuration(design_directory: str | pathlib.Path) -> dict

   This function reads the configuration file found in the design directory.

   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: Configuration dictionary.
   :rtype: config_dictionary(dict)


.. py:function:: create_setup_py_from_config_json(design_directory: str | pathlib.Path) -> None

   This function creates a setup.py file from the config.json file found in the design directory.

   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None
