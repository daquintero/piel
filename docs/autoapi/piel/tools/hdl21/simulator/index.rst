:py:mod:`piel.tools.hdl21.simulator`
====================================

.. py:module:: piel.tools.hdl21.simulator


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.hdl21.simulator.configure_ngspice_simulation
   piel.tools.hdl21.simulator.configure_operating_point_simulation
   piel.tools.hdl21.simulator.configure_transient_simulation
   piel.tools.hdl21.simulator.run_simulation



.. py:function:: configure_ngspice_simulation(run_directory: piel.config.piel_path_types = '.')

   This function configures the NGSPICE simulation for the circuit and returns a simulation class.

   :param run_directory: Directory where the simulation will be run
   :type run_directory: piel_path_types

   :returns: Configured NGSPICE simulation options
   :rtype: simulation_options


.. py:function:: configure_operating_point_simulation(testbench: hdl21.Module, **kwargs)

   This function configures the DC operating point simulation for the circuit and returns a simulation class.

   :param testbench: HDL21 testbench
   :type testbench: Module
   :param \*\*kwargs: Additional arguments to be passed to the operating point simulation such as name.

   :returns: HDL21 simulation class
   :rtype: Simulation


.. py:function:: configure_transient_simulation(testbench: hdl21.Module, stop_time_s: float, step_time_s: float, **kwargs)

   This function configures the transient simulation for the circuit and returns a simulation class.

   :param testbench: HDL21 testbench
   :type testbench: Module
   :param stop_time_s: Stop time of the simulation in seconds
   :type stop_time_s: float
   :param step_time_s: Step time of the simulation in seconds
   :type step_time_s: float
   :param \*\*kwargs: Additional arguments to be passed to the transient simulation

   :returns: HDL21 simulation class
   :rtype: Simulation


.. py:function:: run_simulation(simulation: hdl21.sim.Sim, simulator_name: Literal[ngspice] = 'ngspice', simulation_options: Optional[vlsirtools.spice.SimOptions] = None, to_csv: bool = True)

   This function runs the transient simulation for the circuit and returns the results.

   :param simulation: HDL21 simulation class
   :type simulation: h.sim.Sim
   :param simulator_name: Name of the simulator
   :type simulator_name: Literal["ngspice"]
   :param simulation_options: Simulation options
   :type simulation_options: Optional[vsp.SimOptions]
   :param to_csv: Whether to save the results to a csv file
   :type to_csv: bool

   :returns: Simulation results
   :rtype: results


