# # Further Analogue-Enhanced Cosimulation including `SPICE`

# This example demonstrates the modelling of multi-physical component interconnection and system design.

from gdsfactory.components import mzi2x2_2x2_phase_shifter
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

# So this is very cool, we have our device model giving us electrical data when connected to the geometrical design parameters. What effect does half that resistance have on the driver though? We need to first create a SPICE model of our circuit. One of the main complexities now is that we need to create a mapping between our component models and `PySpice` which is dependent on our device model extraction. Another functionality we might desire is to validate physical electrical connectivity by simulating the circuit accordingly.

# ## Extracting the SPICE circuit and assigning model parameters

# We will exemplify how `piel` microservices enable the extraction and configuration of the SPICE circuit. This is done by implementing a SPICE netlist construction backend to the circuit composition functions in `sax`, and is composed in a way that is then integrated into `PySpice` or any SPICE-based solver through the `VLSIR` `Netlist`.
#
# The way this is achieved is by extracting all the `instances`, `connections`, `ports`, `models` which is essential to compose our circuit using our `piel` SPICE backend solver. It follows a very similar principle to all the other `sax` based circuit composition tools, which are very good.

our_resistive_heater_netlist = our_resistive_heater().get_netlist(
    allow_multiple=True, exclude_port_types="optical"
)
# our_resistive_mzi_2x2_2x2_phase_shifter_netlist = our_resistive_mzi_2x2_2x2_phase_shifter.get_netlist(exclude_port_types="optical")
# our_resistive_heater_netlist

# We might want to extract our connections of our gdsfactory netlist, and convert it to names that can directly form part of a SPICE netlist. However, to do this we need to assign what type of element each component each gdsfactory instance is. We show an algorithm that does the following in order to construct our SPICE netlist:
#
# 1. Extract all instances and required models from the netlist
# 2. Verify that the models have been provided. Each model describes the type of component this is, how many ports it requires and so on.
# 3. Map the connections to each instance port as part of the instance dictionary.
#
# `piel` does this for you already:

our_resistive_heater_spice_netlist = piel.gdsfactory_netlist_to_spice_netlist(
    our_resistive_heater_netlist
)
our_resistive_heater_spice_netlist

# This will allow us to create our SPICE connectivity accordingly because it is in a suitable netlist format. Each of these components in this netlist is some form of an electrical model or component. We start off from our instance definitions. They are in this format:

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
#  'connections': [('straight_1,e1', 'taper_1,e2'),
#   ('straight_1,e2', 'taper_2,e2')],
#  'spice_nets': {'e1': 'straight_1__e1___taper_1__e2',
#   'e2': 'straight_1__e2___taper_2__e2'},
#  'spice_model': <function piel.models.physical.electronic.spice.resistor.basic_resistor()>}
# ```

# We know its connectivity from they key name. In this case, `straight_1` is connected to `taper1` and `taper2`. We know that it is a resistive element, so that means that the SPICE netlist will be in the form:
# ```
# R<SOMEID>
# ```
#
# We can compose our SPICE using PySpice using the models we have provided. The final circuit can be extracted accordingly:

spice_circuit = piel.spice_netlist_to_pyspice_circuit(
    spice_netlist=our_resistive_heater_spice_netlist
)
print(spice_circuit)

# ```
# .title straight_heater_metal_s_b8a2a400
# R0 straight_1__e1___taper_1__e2 straight_1__e2___taper_2__e2 10kOhm
# R2 straight_1__e2___taper_2__e2 taper_2__e1___via_stack_2__e1 10kOhm
# ```

# We can now simulate so much more. Note that this API is WIP.

spice_circuit

# We can extract the electrical components of our heater implementation on its own first.

# ## `PySpice` Integration

# We have seen in the previous example how to integrate digital-driven data with photonic circuit steady-state simulations. However, this is making a big assumption: whenever digital codes are applied to photonic components, the photonic component responds immediately. We need to account for the electrical load physics in order to perform more accurate simulation models of our systems.
#
# `piel` provides a set of basic models for common photonic loads that integrates closely with `gdsfactory`. This will enable the multi-domain co-design we like when integrated with all the open-source tools we have previously exemplified. You can find the list and definition of the provided models in:
#

# ### Creating our Stimulus

# Let's first look into how to map a numpy data into a SPICE waveform. The interconnect of electrical stimulus will be done specifically through standard numpy arrays. One of the main complexities of inputting the SPICE signal information is that SPICE solvers are designed to input specific types of signals such as sine waves, step responses, pulses, squares, sawtooth waves with parameters defined for those signals. However, we might want to explore how custom signals affect our circuit, say in a particular modulation regime. Adding non-standard waves is custom behaviour in some SPICE solvers. However, this can be done in the form of a "Piecewise Linear Controlled Source" as in the 12.2.7 section of the [NGSpice documentation](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf.). One of the complexities is that `PySpice` is not using the latest version of NGSPice. For the sake of simplicity, we will start with the pre-built signal sources.

# ### Mixed-Signal Electronic Photonic Simulation Methodology

# We can now simulate SPICE-based circuits alongside our defined layout electrical models. This allows us the power to implement our own sources, and our own components in a more complete manner. However, this does not fundamentally solve the problem we have on multiple time domain simulations. We have timing data from digital sources and analog electronic sources. If you have delayed photonic signals, then we also have timing data independently in the photonic domain. This leads to a major time-synchronisation issue. It is in this type of problem structure that a microservice implementation again comes to the rescue.
#
# For example, digital signals are particularly valuable and useful when considering steady-state signal propagation. Analogue signals are particularly interesting in the transition between electronic states, as the rise-times and signal-shapes will be determined by RC constants primarily. In the midst of all of this, photonic signals can change independently. There is a direct control from the electronics to the unitary of the component, and at discretised points in time the unitary can be computed. For the period of discrete time this signal is valid, the unitary can be computed. The accuracy desired is just determined by the discretization time. In that period of time, there is a linear relationship between optical signal inputs, and outputs.
#
# As an approximation, if the signals are switching between logic levels, then it is reasonable to compute the analog rise-time and fall-time signals accordingly and operate on them linearly if the digital clock period is high. However, if the logic levels in which the signal is switching is

#
