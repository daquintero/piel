from gdsfactory.components import mzi2x2_2x2_phase_shifter, straight
from gdsfactory.components.straight_heater_metal import straight_heater_metal
import functools
import sax

# Defines the resistance parameters
our_resistor = functools.partial(straight_heater_metal, ohms_per_square=2)

our_resistive_heated_waveguide = functools.partial(
    straight_heater_metal, ohms_per_square=2
)

our_resistive_mzi_2x2_2x2_phase_shifter = mzi2x2_2x2_phase_shifter(
    straight_x_top=our_resistive_heated_waveguide,
)

our_electrical_gdsfactory_netlist = our_resistive_heated_waveguide().get_netlist(
    exclude_port_types="optical"
)

"""One of the main complexities of integrating the SPICE netlist is determining the exact instance connectivity. One
of the main complexities is that in `GDSFactory` many of these electrical components are provided in the form of
cross-sectional paths. This means that we need to find a way to extract connectivity that can be translated from
these fundamental structures.

The first step is that all `GDSFactory` components need to have electrical pins attached to them too, or they can not
be extracted accordingly. It would be ideal for this to be performed as part of a subroutine in the layout generation
scheme, like for the photonic components.

This means we have to add ports onto the cross-sections for particular instantiations of these sections.
 """
straight().show()
print(straight().get_netlist())
# We can see the optical ports in layers 1/10

our_resistive_heated_waveguide().show()
# print(our_resistive_heated_waveguide())
# print(our_electrical_gdsfactory_netlist["instances"].keys())
# We can only see optical ports, and one set of electrical ports, but this is not the full connectivity.
# This means that we need to add the ports from the path.

"""See this basic example in the `gdsfactory` `straight` definition. This means we need to do this for all cross
sections however they are implemented. All instantiations of the cross-section use this straight component definition
so we need to include in some form electrical connectivity accordingly.

    p = gf.path.straight(length=length, npoints=npoints)
    x = gf.get_cross_section(cross_section, **kwargs)

    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)

We can test with an electrical port:
"""
straight(cross_section="metal1").show()

"""This does not add any pins to the electrical component. Note that we need to have a function to add them in order
to extract our electrical netlist connectivity. It would be reasonable, given that the electrical connectivity can be
fanned out, that the whole metal path has pins since geometrical connectivity can be performed from all directions
without DC loss. Now, on the RF side this is a different game, but for the sake of simple connectivity this can be
done. Note that we still have to apply our device parameters in between. The problem becomes defining an input or output, because there is none, there is only both for fundamental physical connectivity."""

print(sax.get_required_circuit_models(our_electrical_gdsfactory_netlist))
# # ['via_stack'] # Currently this is being returned, but this is not an accurate circuit model.
