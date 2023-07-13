"""
These functions map a particular model, with an instance representation that corresponds to the given netlist
connectivity, and returns a PySpice representation of the circuit. This function will be called after parsing the
circuit netlist accordingly, and creating a mapping from the instance definitions to the fundamental components.
"""
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_kOhm


def add_basic_resistor(
    circuit: Circuit,
    instance_id: int,
    input_node: str,
    output_node: str,
):
    """
    SPICE Resistor Structure

    See Mike Smith “WinSpice3 User’s Manual” 25 October, 1999

    .. code-block::

        RXXXXXXX N1 N2 <VALUE> <MNAME> <L=LENGTH> <W=WIDTH> <TEMP=T>

    Where the terminals are:

    .. code-block::

        N1 = the first terminal
        N2 = the second terminal
        <VALUE> = resistance in ohms.
        <MNAME> = name of the model used (useful for semiconductor resistors)
        <L=LENGTH> = length of the resistor (useful for semiconductor resistors)
        <W=WIDTH> = width of the resistor (useful for semiconductor resistors)
        <TEMP=T> = temperature of the resistor in Kelvin (useful in noise analysis and
        semiconductor resistors)

    An example is:

    .. code-block::

        RHOT n1 n2 10k TEMP=500
    """
    circuit.R(instance_id, input_node, output_node, u_kOhm(10))
    return circuit
