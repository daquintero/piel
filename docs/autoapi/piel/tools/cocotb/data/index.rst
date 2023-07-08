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

   # TODO DOCS


.. py:function:: read_simulation_data(file_path)

   This function returns a Pandas dataframe that contains all the simulation data outputted from the simulation run.


.. py:function:: simple_plot_simulation_data(simulation_data: pandas.DataFrame)
