"""The way each of these models should work is that they use the settings from the `gdsfactory` component,
to create a parametric SPICE directive. """


def a(settings) -> str:
    """
    This function takes in the settings from a gdsfactory component, some connectivity node translated directly from the gdsfactory netlist.
    """
    pass


"""
SPICE capacitor model:

.. code-block::
    CXXXXXXX N+ N- VALUE <IC=INCOND>

Where the parameters are:

.. code-block::
    N+ = the positive terminal
    N- = the negative terminal
    VALUE = capacitance in farads
    <IC=INCOND> = starting voltage in a simulation
"""
