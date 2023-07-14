"""The way the construction of the SPICE models is implemented in ``piel`` is also microservice-esque. Larger
circuits are constructed out of smaller building blocks. This means that simulation configurations and so on are
constructed upon a provided initial circuit, and the SPICE directives are appended accordingly.

As mentioned previously, ``piel`` creates circuits based on fundamental SPICE because this opens the possibility to
large scale integration of these circuit models on different SPICE solvers, including proprietary ones as long as the
SPICE circuit can be written to particular directories. However, due to the ease of circuit integration,
and translation that ``PySpice`` provides, it's API can also be used to implement parametrised functionality,
although it is easy to export the circuit as a raw SPICE directive after composition.

This composition tends to be implemented in a `SubCircuit` hierarchical implementation, as this allows for more modularisation of the netlisted devices. Another aspect of complexity is that `PySpice` is class-based composed, so that means that functions define class definitions and return them, or instantiate default ones.

Let's assume that we can get an extracted SPICE netlist of our circuit, that includes all nodes, and component
circuit definitions. This could then be simulated accordingly for the whole circuit between inputs and outputs. This
would have to be constructed out of component models and a provided netlist in a similar fashion to ``SAX``. """
from PySpice.Spice.Netlist import Circuit

__all__ = ["create_pyspice_circuit", "write_raw_spice_to_pyspice_circuit"]


def create_pyspice_circuit(circuit_name: str = "c") -> Circuit:
    """
    This function returns a generic PySPICE circuit. This is useful for adding SPICE directives to a circuit.

    Args:
        circuit_name (str): Name of the circuit

    Returns:
        Circuit: PySpice circuit
    """
    circuit = Circuit(circuit_name)
    return circuit


def write_raw_spice_to_pyspice_circuit(circuit: Circuit, raw_spice: str) -> Circuit:
    """
    This function returns a PySpice circuit including the raw SPICE text. This is useful for adding SPICE directives
    to a circuit.

    Args:
        circuit (Circuit): PySpice circuit
        raw_spice (str): Raw SPICE text

    Returns:
        Circuit: PySpice circuit including the raw SPICE text
    """
    circuit.raw_spice += raw_spice
    return circuit
