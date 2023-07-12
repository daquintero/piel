def instance_to_pyspice(component_model):
    """
    This function maps a particular model, with an instance representation that corresponds to the given netlist
    connectivity, and returns a PySpice representation of the circuit. This function will be called after parsing the
    circuit netlist accordingly, and creating a mapping from the instance definitions to the fundamental components.

    Args:
        component_model(func): Function that represents a SPICE component with the given parameters.
    """
