# `gdsfactory` - SPICE `Netlist` - `VLSIR`

There are many ways in which we might desire to integrate `gdsfactory` with `SPICE`. One particular one is to be able to extract an electrical netlist from a photonic circuit, simulate the performance, and connect it to a circuit driver previously. This requires in some form extracting the port interconnection from the `gdsfactory` netlist, and then connecting this as part of an electronic component. This will allow us to integrate with the `ngspice` and `xyce` simulators, and in the future with Cadence and related tools.

## Integration Methodology

One of the main complexities of integrating the SPICE netlist is determining the exact instance connectivity. One of the main complexities is that in `GDSFactory` many of these electrical components are provided in the form of cross-sectional paths. This means that we need to find a way to extract connectivity that can be translated from these fundamental structures.

This implementation follows the principle that we can build SPICE netlists out of the provided `GDSFactory` netlists which are port dependent. This netlist is constructed using port interconnectivity, but it is necessary to define correctly the layout of the circuit in order to get the correct interconnection and accurately generate SPICE models. To do this, we need to provide models in a very close implementation to `sax`. We cannot extract a SPICE interconnect without using models and using the `Netlist` package from Dan Friedman means we are ensuring future compatibility to `VLSIR` and so on. We can then perform a LVS check with the actual layout using further tools as discussed below.


### LVS Tools Integration

The first step is that all `GDSFactory` components need to have electrical pins attached to them too, or they can not be extracted accordingly. It would be ideal for this to be performed as part of a subroutine in the layout generation scheme, like for the photonic components.

In order to extract physical electrical connectivity, then we need to make sure the pin connectivity is included in between metal layers, and that VIAs allow for connectivity between layers. This needs to conform to the standard in tools such as Cadence. In this tool, the pin process is a different purpose than the drawing layer. I can see in the generic PDK the WG_PIN layer it was already set to purpose 10 so maybe stick with it for the standard.
