# `pyspice` - `gdsfactory`

There are many ways in which we might desire to integrate `gdsfactory` with `pyspice`. One particular one is to be able to extract an electrical netlist from a photonic circuit, simulate the performance, and connect it to a circuit driver previously. This requires in some form extracting the port interconnection from the `gdsfactory` netlist, and then connecting this as part of an electronic component. This will allow us to integrate with the `ngspice` and `xyce` simulators, and in the future with Cadence and related tools.
