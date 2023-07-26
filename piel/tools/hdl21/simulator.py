import hdl21 as h
import hdl21.sim as hs


def configure_transient_simulation(
    circuit: h.Module,
    stop_time_s: float,
    step_time_s: float,
):
    """
    This function configures the transient simulation for the circuit.
    """

    @hs.sim
    class Simulation:
        """Custom Simulation class"""

        # Simulation Stimulus
        op = hs.Op()  # DC Operating Point Analysis
        a_path = "."
        include_that_path = hs.Include(a_path)

    class Simulatioan:
        tb = circuit
        transient_tb = hs.Tran(stop_time_s=stop_time_s, step_time_s=step_time_s)
        save_all = hs.Save(hs.SaveMode.ALL)

    return Simulation


# def configure_ngspice_simulator(
#     circuit,
# ) -> Circuit:
#     """
#     This function configures the NgSpice simulator to be used for the simulation.
#
#     Example usage below:
#
#     .. code-block::
#
#         from PySpice.Spice.Netlist import Circuit
#         from PySpice.Unit import u_kOhm
#         from piel.tools.hdl21.simulator import configure_ngspice_simulator
#         +
#         circuit = Circuit('Resistor')
#         circuit.R(1, 'input', 'output', u_kOhm(10))
#         circuit = configure_ngspice_simulator(circuit)
#
#
#     Args:
#         circuit (Circuit): PySpice circuit
#
#     Returns:
#         Circuit: PySpice circuit with NgSpice simulator configured
#     """
#     ngspice = NgSpiceShared.new_instance()
#     circuit = circuit.simulator(
#         temperature=25,
#         nominal_temperature=25,
#         simulator="ngspice-shared",
#         ngspice_shared=ngspice,
#     )
#     return circuit
#
#
# def configure_transient_simulation(circuit: Circuit) -> Circuit:
#     """
#     This function configures the transient simulation to be used for the simulation.
#
#     Example usage below:
#
#     .. code-block::
#
#         from PySpice.Spice.Netlist import Circuit
#         from PySpice.Unit import u_kOhm
#         from piel.tools.hdl21.simulator import configure_transient_simulation
#         +
#         circuit = Circuit('Resistor')
#         circuit.R(1, 'input', 'output', u_kOhm(10))
#         circuit = configure_transient_simulation(circuit)
#     """
#     # circuit = circuit.transient(step_time=1 @ u_us, end_time=500 @ u_us)
#     return circuit
