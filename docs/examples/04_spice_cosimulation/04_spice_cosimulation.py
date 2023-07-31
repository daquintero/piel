# # Further Analogue-Enhanced Cosimulation including `SPICE`

# This example demonstrates the modelling of multi-physical component interconnection and system design.

from gdsfactory.components import mzi2x2_2x2_phase_shifter
import hdl21 as h
import pandas as pd
import piel
import sys

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

# ![our_short_resistive_heater](../../_static/img/examples/04_spice_cosimulation/our_short_resistive_heater.PNG)

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

our_resistive_heater_spice_netlist = piel.gdsfactory_netlist_with_hdl21_generators(
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
# {'straight_1': Instance(name=straight_1 of=GeneratorCall(gen=straight)),
#  'taper_1': Instance(name=taper_1 of=GeneratorCall(gen=taper)),
#  'taper_2': Instance(name=taper_2 of=GeneratorCall(gen=taper)),
#  'via_stack_1': Instance(name=via_stack_1 of=GeneratorCall(gen=via_stack)),
#  'via_stack_2': Instance(name=via_stack_2 of=GeneratorCall(gen=via_stack))}
# ```

# Note that each component is mapped into `hdl21` according to the same structure and names as in the `gdsfactory` netlist, if you have defined your generator components correctly. Note that the unconnected ports need to be exposed for proper SPICE composition.

our_resistive_heater_circuit.ports

# ```python
# {'e1': Signal(name='e1', width=1, desc=None),
#  'e2': Signal(name='e2', width=1, desc=None),
#  'via_stack_1__e1': Signal(name='via_stack_1__e1', width=1, desc=None),
#  'via_stack_1__e2': Signal(name='via_stack_1__e2', width=1, desc=None),
#  'via_stack_1__e4': Signal(name='via_stack_1__e4', width=1, desc=None),
#  'via_stack_2__e2': Signal(name='via_stack_2__e2', width=1, desc=None),
#  'via_stack_2__e3': Signal(name='via_stack_2__e3', width=1, desc=None),
#  'via_stack_2__e4': Signal(name='via_stack_2__e4', width=1, desc=None)}
# ```

# Same for the signals

our_resistive_heater_circuit.signals

# ```python
# {'via_stack_1_e3': Signal(name='via_stack_1_e3', width=1, desc=None),
#  'via_stack_2_e1': Signal(name='via_stack_2_e1', width=1, desc=None)}
# ```

# We can explore the models we have provided too:

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

# One of the main complexities of this translation is that for the SPICE to be generated, the network has to be valid. Sometimes, in direct `gdsfactory` components, there could be incomplete or undeclared port networks. This means that for the SPICE to be generated, we have to fix the connectivity in some form, and means that there might not be a direct translation from `gdsfactory`. This is inherently related to the way that the construction of the netlists are generated. This means that to some form, we need to connect the unconnected ports in order for the full netlist to be generated. The complexity is in components such as the via where there are four ports to it on each side. SPICE would treat them as four different inputs that need to be connected.

# Now let's extract the SPICE for our heater circuit:

h.netlist(our_resistive_heater_circuit, sys.stdout, fmt="spice")

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
#
# .SUBCKT Taper__
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
#
# .SUBCKT ViaStack__
# + e1 e2 e3 e4
# * No parameters
#
# rr1
# + e1 e2
# + 1000
# * No parameters
#
#
# rr2
# + e2 e3
# + 1000
# * No parameters
#
#
# rr3
# + e3 e4
# + 1000
# * No parameters
#
#
# rr4
# + e4 e1
# + 1000
# * No parameters
#
#
# .ENDS
#
# .SUBCKT straight_heater_metal_s_b8a2a400
# + e1 e2 via_stack_1__e1 via_stack_1__e2 via_stack_1__e4 via_stack_2__e2 via_stack_2__e3 via_stack_2__e4
# * No parameters
#
# xstraight_1
# + e1 e2
# + Straight__
# * No parameters
#
#
# xtaper_1
# + via_stack_1_e3 e1
# + Taper__
# * No parameters
#
#
# xtaper_2
# + via_stack_2_e1 e2
# + Taper__
# * No parameters
#
#
# xvia_stack_1
# + via_stack_1__e1 via_stack_1__e2 via_stack_1_e3 via_stack_1__e4
# + ViaStack__
# * No parameters
#
#
# xvia_stack_2
# + via_stack_2_e1 via_stack_2__e2 via_stack_2__e3 via_stack_2__e4
# + ViaStack__
# * No parameters
#
#
# .ENDS
#
# ```

# So this seems equivalent to the gdsfactory component representation. We can now continue to implement our SPICE simulation.

# ## `SPICE` Integration

# We have seen in the previous example how to integrate digital-driven data with photonic circuit steady-state simulations. However, this is making a big assumption: whenever digital codes are applied to photonic components, the photonic component responds immediately. We need to account for the electrical load physics in order to perform more accurate simulation models of our systems.
#
# `piel` provides a set of basic models for common photonic loads that integrates closely with `gdsfactory`. This will enable the multi-domain co-design we like when integrated with all the open-source tools we have previously exemplified. You can find the list and definition of the provided models in:
#
# Currently, these models do not physically represent electrically the components yet. We will do this later.

piel_hdl21_models = piel.models.physical.electronic.get_default_models()
piel_hdl21_models

# We can extract the SPICE out of each model

example_straight_resistor = piel_hdl21_models["straight"]()
h.netlist(example_straight_resistor, sys.stdout, fmt="spice")


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

# ### Creating our Stimulus

# Let's first look into how to map a numpy data into a SPICE waveform. We will create a testbench using the `hdl21` interface. So the first thing is that we need to add our *stimulus* sources, or otherwise where would our pulses come from. We need to construct this into our testbench module. This means we need to generate the connectivity of our signal sources in relation to the ports of our circuit. We create a custom testbench module that we will use to perform this simulation. This needs to contain also our voltage sources for whatever test that we would be performing.


@h.module
class OperatingPointTb:
    """# Basic Extracted Device DC Operating Point Testbench"""

    VSS = h.Port()  # The testbench interface: sole port VSS - GROUND
    VDC = h.Vdc(dc=1)(n=VSS)  # A DC voltage source

    # Our component under test
    example_straight_resistor.e1 = VDC.p
    example_straight_resistor.e2 = VSS


# In this basic test, we can analyse the DC operating point relationship for this circuit. Now, in order to make these type of simulations more automated to run at scale, `piel` provides some wrapper functions that can be parameterised with most common simulation parameters:

# #### A Simple DC Operating Point Simulation

simple_operating_point_simulation = piel.configure_operating_point_simulation(
    testbench=OperatingPointTb, name="simple_operating_point_simulation"
)
simple_operating_point_simulation

# ```python
# Sim(tb=Module(name=OperatingPointTb), attrs=[Op(name='operating_point_tb'), Save(targ=<SaveMode.ALL: 'all'>)], name='Simulation')
# ```

# We can now run the simulation using `ngpsice`. Make sure you have it installed, although this will be automatic in the *IIC-OSIC-TOOLS* environment:

results = piel.run_simulation(simulation=simple_operating_point_simulation)
results

# ```python
# SimResult(an=[OpResult(analysis_name='operating_point_tb', data={'v(xtop.vdc_p)': 1.0, 'i(v.xtop.vvdc)': 0.0})])
# ```

# We can access the data as a dictionary too:

results.an[0].data

# ```python
# {'v(xtop.vdc_p)': 1.0, 'i(v.xtop.vvdc)': 0.0}
# ```

# #### A Simple Transient Simulation
#

# Let's assume we want to simulate in time how a pulse propagates through our circuit. A resistor on its own will have a linear relationship with the pulse, which means we should see how the current changes from the input pulse in time accordingly. For clarity, we will make a new testbench, even if there are ways to combine them.


@h.module
class TransientTb:
    """# Basic Extracted Device DC Operating Point Testbench"""

    VSS = h.Port()  # The testbench interface: sole port VSS - GROUND
    VDC = h.Vpulse(
        delay=1 * h.prefix.m,
        v1=0 * h.prefix.m,
        v2=1000 * h.prefix.m,
        period=10 * h.prefix.m,
        rise=1 * h.prefix.m,
        fall=1 * h.prefix.m,
        width=10 * h.prefix.m,
    )(
        n=VSS
    )  # A configured voltage pulse source

    # Our component under test
    example_straight_resistor.e1 = VDC.p
    example_straight_resistor.e2 = VSS


# Again we use a simple `piel` wrapper:

simple_transient_simulation = piel.configure_transient_simulation(
    testbench=TransientTb,
    stop_time_s=100e-3,
    step_time_s=1e-4,
    name="simple_transient_simulation",
)
simple_transient_simulation

# ```python
# Sim(tb=Module(name=TransientTb), attrs=[Tran(tstop=0.1*UNIT, tstep=0.0001*UNIT, name='transient_tb')], name='Simulation')
# ```

piel.run_simulation(simple_transient_simulation, to_csv=True)

# When you run the simulation using the `piel` `run_simulation` command, there is a `to_csv` flag that allows us to save the data and access it afterwards. We access the transient simulation in Pandas accordingly:

transient_simulation_results = pd.read_csv("TransientTb.csv")
transient_simulation_results.iloc[120:140]

# |     | Unnamed: 0 |    time | v(xtop.vdc_p) | i(v.xtop.vvdc) |
# |----:|-----------:|--------:|--------------:|--------------:|
# | 120 |        120 | 0.01075 |            1  |            0  |
# | 121 |        121 | 0.01085 |            1  |            0  |
# | 122 |        122 | 0.01095 |            1  |            0  |
# | 123 |        123 | 0.01105 |         0.05  |            0  |
# | 124 |        124 | 0.01115 |         0.15  |            0  |
# | 125 |        125 | 0.01125 |         0.25  |            0  |
# | 126 |        126 | 0.01135 |         0.35  |            0  |
# | 127 |        127 | 0.01145 |         0.45  |            0  |
# | 128 |        128 | 0.01155 |         0.55  |            0  |
# | 129 |        129 | 0.01165 |         0.65  |            0  |
# | 130 |        130 | 0.01175 |         0.75  |            0  |
# | 131 |        131 | 0.01185 |         0.85  |            0  |
# | 132 |        132 | 0.01195 |         0.95  |            0  |
# | 133 |        133 | 0.012   |            1  |            0  |
# | 134 |        134 | 0.01201 |            1  |            0  |
# | 135 |        135 | 0.01203 |            1  |            0  |
# | 136 |        136 | 0.01207 |            1  |            0  |
# | 137 |        137 | 0.01215 |            1  |            0  |
# | 138 |        138 | 0.01225 |            1  |            0  |
# | 139 |        139 | 0.01235 |            1  |            0  |

# We can plot our simulation data accordingly:

simple_transient_plot = piel.visual.plot_simple_multi_row(
    data=transient_simulation_results,
    x_axis_column_name="time",
    row_list=[
        "v(xtop.vdc_p)",
        "i(v.xtop.vvdc)",
    ],
    y_axis_title_list=["v(xtop.vdc_p)", "i(v.xtop.vvdc)", "o4 Phase"],
)
# simple_transient_plot.savefig("simple_transient_plot.png")

# ![simple_transient_plot](../../_static/img/examples/04_spice_cosimulation/simple_transient_plot.PNG)

# #### Automation

# Now, these transient simulations are something you might want to very configure depending on the type of signals that you might want to verify. However, we can provide some basic parameterised simple functions such as step responses and so on. So instead of having to write everything above, you can also just run the following, for example:
