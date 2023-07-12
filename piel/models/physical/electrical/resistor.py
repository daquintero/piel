def resistor_spice():
    """
    SPICE Resistor Structure

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
    pass
