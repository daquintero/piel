:py:mod:`piel.tools.cocotb.data`
================================

.. py:module:: piel.tools.cocotb.data

.. autoapi-nested-parse::

   This file contains a range of functions used to read, plot and analyse cocotb simulations in a data-flow standard as suggested



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.cocotb.data.get_simulation_output_files_from_design
   piel.tools.cocotb.data.read_simulation_data
   piel.tools.cocotb.data.simple_plot_simulation_data



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.cocotb.data.get_simulation_output_files


.. py:data:: get_simulation_output_files



.. py:function:: get_simulation_output_files_from_design(design_directory: piel.config.piel_path_types, extension: str = 'csv')

   This function returns a list of all the simulation output files in the design directory.

   :param design_directory: The path to the design directory.
   :type design_directory: piel_path_types

   :returns: List of all the simulation output files in the design directory.
   :rtype: output_files (list)


.. py:function:: read_simulation_data(file_path: piel.config.piel_path_types)

   This function returns a Pandas dataframe that contains all the simulation data outputted from the simulation run.

   :param file_path: The path to the simulation data file.
   :type file_path: piel_path_types

   :returns: The simulation data in a Pandas dataframe.
   :rtype: simulation_data (pd.DataFrame)


.. py:function:: simple_plot_simulation_data(simulation_data: pandas.DataFrame)
