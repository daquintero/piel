SPICE Integration
=================

The implementation mechanism is to provide component models that include
the raw interconnect based on similar ``gdsfactory`` port naming and
matching. This will allow us to design netlists that can be closely
mapped into a SPICE solver, directly from ``gdsfactory``. This may
eventually be interconnected through
`VLSIR <https://github.com/Vlsir/Vlsir>`__.

Note that for a particular ``SPICE`` implementation, more than
time-domain simulations can be performed as well. This is why ``piel``
provides functionality to construct simulations to perform these
multi-domain simulations and construct these systems from component
model primitives.


Model Composition with ``hdl21``
---------------------------------

These functions map a particular model, with an instance representation that corresponds to the given netlist
connectivity, and returns a SPICE representation of the circuit in the form of a ``hdl21`` structure. This function
will be called after parsing the circuit netlist accordingly, and creating a mapping from the instance definitions to
the fundamental components.

However, each model may be reasonable for it to be a parametric generator based on the settings of the gdsfactory
instance. Note that in order to not have to assign the directionality of ports it could be reasonable to use signals
instead of ports. We generate this instance based on the ``gdsfactory`` port keys, although the assignment of which
is an input or output is essential based on the initially parsed netlist. A generator follows the following syntax
and the output module can be connected as part of a larger instantiation accordingly.

.. code-block::

    import hdl21 as h
    @h.generator
    def MyFirstGenerator(params: MyParams) -> h.Module:
        # A very exciting first generator function
        m = h.Module()
        m.i = h.Input(width=params.w)
        return m

The models provided are ``hdl21`` generators based on parameters that are inputted from the instance settings for the
defined set of parameters and we use the construct it from the nested dictionary:

.. code-block::
@h.paramclass
    class Outer:
        inner = h.Param(dtype=Inner, desc="Inner fields")
        f = h.Param(dtype=float, desc="A float", default=3.14159)

    # Create from a (nested) dictionary literal
    d1 = {"inner": {"i": 11}, "f": 22.2}
    o = Outer(**d1)

So this allows us to return instance models that compose our circuit.
