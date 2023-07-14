from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def configure_ngspice_simulator(
    circuit: Circuit,
) -> Circuit:
    """
    This function configures the NgSpice simulator to be used for the simulation.

    Example usage below:

    .. code-block::

        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_kOhm
        from piel.tools.pyspice.simulator import configure_ngspice_simulator
        +
        circuit = Circuit('Resistor')
        circuit.R(1, 'input', 'output', u_kOhm(10))
        circuit = configure_ngspice_simulator(circuit)


    Args:
        circuit (Circuit): PySpice circuit

    Returns:
        Circuit: PySpice circuit with NgSpice simulator configured
    """
    ngspice = NgSpiceShared.new_instance()
    circuit = circuit.simulator(
        temperature=25,
        nominal_temperature=25,
        simulator="ngspice-shared",
        ngspice_shared=ngspice,
    )
    return circuit


def configure_transient_simulation(circuit: Circuit) -> Circuit:
    """
    This function configures the transient simulation to be used for the simulation.

    Example usage below:

    .. code-block::

        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_kOhm
        from piel.tools.pyspice.simulator import configure_transient_simulation
        +
        circuit = Circuit('Resistor')
        circuit.R(1, 'input', 'output', u_kOhm(10))
        circuit = configure_transient_simulation(circuit)
    """
    # circuit = circuit.transient(step_time=1 @ u_us, end_time=500 @ u_us)
    return circuit
