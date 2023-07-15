# # Further Analogue-Enhanced Cosimulation including `SPICE`

# This example demonstrates the modelling of multi-physical component interconnection and system design.

import sys
from gdsfactory.components import mzi2x2_2x2_phase_shifter
import hdl21 as h
import piel

# ## Start from `gdsfactory`

# Let us begin from the `mzi2x2_2x2_phase_shifter` circuit we have been analysing so far in our examples, but this time, we will explore further aspects of the electrical interconnect, and try to extract or extend the simulation functionality. Note that this flattened netlist is the lowest level connectivity analysis, for very computationally complex extraction. Explore the electrical circuit connectivity using the keys below.

# + active=""
# mzi2x2_2x2_phase_shifter_netlist_flat = mzi2x2_2x2_phase_shifter().get_netlist_flat(
#     exclude_port_types="optical"
# )
# mzi2x2_2x2_phase_shifter_netlist_flat.keys()
# -

# Note that this netlist just gives us electrical ports and connectivity for this component.

# + active=""
# mzi2x2_2x2_phase_shifter_netlist_flat["connections"]
# -

# The most basic model of this phase-shifter device is treating the heating element as a resistive element. We want to explore how this resistor model affects the rest of the photonic design circuitry. You can already note something important if you explored the netlist with sufficient care and followed the previous examples regarding the `straight_x_top` or `sxt` heater component.

mzi2x2_2x2_phase_shifter_netlist = mzi2x2_2x2_phase_shifter().get_netlist(
    exclude_port_types="optical"
)
mzi2x2_2x2_phase_shifter_netlist["instances"]["sxt"]

# ```python
# {'component': 'straight_heater_metal_undercut',
#  'info': {'resistance': None},
#  'settings': {'cross_section': 'strip',
#   'length': 200.0,
#   'with_undercut': False}}
# ```

# So this top heater instance `info` instance definition, it already includes a `resistance` field. However, in the default component definition, it is defined as `None`. Let us give some more details about our circuit, and this would normally be provided by the PDK information of your foundry.

# +
from gdsfactory.components.straight_heater_metal import straight_heater_metal_simple
import functools

# Defines the resistance parameters
our_resistive_heater = functools.partial(
    straight_heater_metal_simple, ohms_per_square=2
)

our_resistive_mzi_2x2_2x2_phase_shifter = mzi2x2_2x2_phase_shifter(
    straight_x_top=our_resistive_heater,
)
our_resistive_mzi_2x2_2x2_phase_shifter
# -

# ![mzi2x2_2x2_phase_shifter](../_static/img/examples/03a_sax_active_cosimulation/mzi2x2_phase_shifter.PNG)

our_resistive_mzi_2x2_2x2_phase_shifter_netlist = (
    our_resistive_mzi_2x2_2x2_phase_shifter.get_netlist(exclude_port_types="optical")
)
our_resistive_mzi_2x2_2x2_phase_shifter_netlist["instances"]["sxt"]

# ```python
# {'component': 'straight_heater_metal_undercut',
#  'info': {'resistance': 1000.0},
#  'settings': {'cross_section': 'strip',
#   'length': 200.0,
#   'ohms_per_square': 2,
#   'with_undercut': False}}
# ```

# Now we have a resistance parameter directly connected to the geometry of our phase shifter topology, which is an incredibly powerful tool. In your models, you can compute or extract resisivity and performance data parameters of your components in order to create SPICE models from them. This is related to the way that you perform your component design. `piel` can then enable you to understand how this component affects the total system performance.
#
# We can make another variation of our phase shifter to explore physical differences.

our_short_resistive_mzi_2x2_2x2_phase_shifter = mzi2x2_2x2_phase_shifter(
    straight_x_top=our_resistive_heater,
    length_x=100,
)
our_short_resistive_mzi_2x2_2x2_phase_shifter.show()
our_short_resistive_mzi_2x2_2x2_phase_shifter.plot_widget()

# ![our_short_resistive_heater](../_static/img/examples/04_multi_domain_interconnect/our_short_resistive_heater.PNG)

# You can also find out more information of this component through:

our_short_resistive_mzi_2x2_2x2_phase_shifter.named_references["sxt"].info

# {'resistance': 500.0}

# So this is very cool, we have our device model giving us electrical data when connected to the geometrical design parameters. What effect does half that resistance have on the driver though? We need to first create a SPICE model of our circuit. One of the main complexities now is that we need to create a mapping between our component models and `hdl21` which is dependent on our device model extraction. Another functionality we might desire is to validate physical electrical connectivity by simulating the circuit accordingly.

# ## Extracting the SPICE circuit and assigning model parameters

# We will exemplify how `piel` microservices enable the extraction and configuration of the SPICE circuit. This is done by implementing a SPICE netlist construction backend to the circuit composition functions in `sax`, and is composed in a way that is then integrated into `hdl21` or any SPICE-based solver through the `VLSIR` `Netlist`.
#
# The way this is achieved is by extracting all the `instances`, `connections`, `ports`, `models` which is essential to compose our circuit using our `piel` SPICE backend solver. It follows a very similar principle to all the other `sax` based circuit composition tools, which are very good.

our_resistive_heater_netlist = our_resistive_heater().get_netlist(
    allow_multiple=True, exclude_port_types="optical"
)
# our_resistive_mzi_2x2_2x2_phase_shifter_netlist = our_resistive_mzi_2x2_2x2_phase_shifter.get_netlist(exclude_port_types="optical")
# our_resistive_heater_netlist

# We might want to extract our connections of our gdsfactory netlist, and convert it to names that can directly form part of a SPICE netlist. However, to do this we need to assign what type of element each component each gdsfactory instance is. We show an algorithm that does the following in order to construct our SPICE netlist:
#
# 1. Parse the gdsfactory netlist, assign the electrical ports for the model. Extract all instances and
# required models from the netlist.
# 2. Verify that the models have been provided. Each model describes the type of
# component this is, how many ports it requires and so on. Create a ``hdl21`` top level module for every gdsfactory
# netlist, this is reasonable as it is composed, and not a generator class. This generates a large amount of instantiated ``hdl21`` modules that are generated from `generators`.
# 3. Map the connections to each instance port as part of the instance dictionary. This parses the connectivity in the ``gdsfactory`` netlist and connects the ports accordingly.
#
# `piel` does this for you already:

our_resistive_heater_spice_netlist = piel.gdsfactory_netlist_with_hdl21_models(
    our_resistive_heater_netlist
)
our_resistive_heater_spice_netlist

# This will allow us to create our SPICE connectivity accordingly because it is in a suitable netlist format using `hdl21`. Each of these components in this netlist is some form of an electrical model or component. We start off from our instance definitions. They are in this format:

our_resistive_heater_netlist["instances"]["straight_1"]

# ```python
# {'component': 'straight',
#  'info': {'length': 320.0,
#   'width': 0.5,
#   'cross_section': 'strip_heater_metal',
#   'settings': {'width': 0.5,
#    'layer': 'WG',
#    'heater_width': 2.5,
#    'layer_heater': 'HEATER'},
#   'function_name': 'strip_heater_metal'},
#  'settings': {'cross_section': 'strip_heater_metal',
#   'heater_width': 2.5,
#   'length': 320.0},
#  'hdl21_model': Generator(name=Straight)}
# ```

# We can compose our SPICE using ``hdl21`` using the models we have provided. The final circuit can be extracted accordingly:

our_resistive_heater_circuit = piel.construct_hdl21_module(
    spice_netlist=our_resistive_heater_spice_netlist
)
our_resistive_heater_circuit.instances

# ```python
# {'straight_1': Instance(name=None of=GeneratorCall(gen=straight)),
#  'taper_1': Instance(name=None of=GeneratorCall(gen=taper)),
#  'taper_2': Instance(name=None of=GeneratorCall(gen=taper)),
#  'via_stack_1': Instance(name=None of=GeneratorCall(gen=via_stack)),
#  'via_stack_2': Instance(name=None of=GeneratorCall(gen=via_stack))}
# ```

# Note that each component is mapped into `hdl21` according to the same structure and names as in the `gdsfactory` netlist, if you have defined your generator components correctly.

our_resistive_heater_circuit.ports

# ```python
# {'e1': Signal(name=None, width=1, desc=None),
#  'e2': Signal(name=None, width=1, desc=None)}
# ```

# Same for the signals

our_resistive_heater_circuit.signals

# ```python
# {'taper_1_e2': Signal(name='taper_1_e2', width=1, desc=None),
#  'taper_2_e2': Signal(name='taper_2_e2', width=1, desc=None),
#  'via_stack_1_e3': Signal(name='via_stack_1_e3', width=1, desc=None),
#  'via_stack_2_e1': Signal(name='via_stack_2_e1', width=1, desc=None)}
# ```

# We can explore the models we have provided too:

piel_hdl21_models = piel.models.physical.electronic.get_default_models()
piel_hdl21_models

# We can extract the SPICE out of each model

h.netlist(piel_hdl21_models["straight"](), sys.stdout, fmt="spice")

# ```spice
# * Anonymous `circuit.Package`
# * Generated by `vlsirtools.SpiceNetlister`
# *
#
# .SUBCKT Straight__
# + e1 e2
# * No parameters
#
# rr1
# + e1 e2
# + 1000
# * No parameters
#
#
# .ENDS
# ```

# We can also extract information for our subinstances. We can also get the netlist of each subinstance using the inherent `hdl21` functionality:

h.netlist(
    our_resistive_heater_circuit.instances["straight_1"].of, sys.stdout, fmt="spice"
)

# ```
# * Anonymous `circuit.Package`
# * Generated by `vlsirtools.SpiceNetlister`
# *
#
# .SUBCKT Straight__
# + e1 e2
# * No parameters
#
# rr1
# + e1 e2
# + 1000
# * No parameters
#
#
# .ENDS
# ```

# Now let's extract the SPICE for our heater circuit:

h.netlist(our_resistive_heater_circuit, sys.stdout, fmt="spice")

# We can now simulate so much more. Note that this API is WIP.

# ## `SPICE` Integration

# We have seen in the previous example how to integrate digital-driven data with photonic circuit steady-state simulations. However, this is making a big assumption: whenever digital codes are applied to photonic components, the photonic component responds immediately. We need to account for the electrical load physics in order to perform more accurate simulation models of our systems.
#
# `piel` provides a set of basic models for common photonic loads that integrates closely with `gdsfactory`. This will enable the multi-domain co-design we like when integrated with all the open-source tools we have previously exemplified. You can find the list and definition of the provided models in:
#

# ### Creating our Stimulus

# Let's first look into how to map a numpy data into a SPICE waveform.

# ### Mixed-Signal Electronic Photonic Simulation Methodology
