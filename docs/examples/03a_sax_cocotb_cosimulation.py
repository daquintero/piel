# # Digital & Photonic Cosimulation with `sax` and `cocotb`

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2
import numpy as np
import piel
import sax

# ## Active MZI 2x2 Phase Shifter

# First, let's look at our actively driven component:

mzi2x2_2x2_phase_shifter().show()
mzi2x2_2x2_phase_shifter().plot_widget()

# ![mzi2x2_2x2_phase_shifter](../_static/img/examples/03a_sax_active_cosimulation/mzi2x2_phase_shifter.PNG)

mzi2x2_2x2_phase_shifter_netlist = mzi2x2_2x2_phase_shifter().get_netlist(
    exclude_port_types="electrical"
)
mzi2x2_2x2_phase_shifter_netlist["instances"].keys()

# ```python
# dict_keys(['bend_euler_1', 'bend_euler_2', 'bend_euler_3', 'bend_euler_4', 'bend_euler_5', 'bend_euler_6', 'bend_euler_7', 'bend_euler_8', 'cp1', 'cp2', 'straight_4', 'straight_5', 'straight_6', 'straight_7', 'straight_8', 'straight_9', 'sxb', 'sxt', 'syl', 'sytl'])
# ```

# From the `mzi2x2_2x2_phase_shifter` component definition, we know that the `sxt` instance in the netlist corresponds to the `straight_heater_metal_simple` actively driven phase shifter in our network.

mzi2x2_2x2_phase_shifter_netlist["instances"]["sxt"]

# So what we do is that if we define an active mode for this waveguide, we can model the network system.

# ## Electronic-to-Phase Mapping
#
# Let us explore first the fundamental relationship between electronic signals to optical phase. When we apply an electronic signal to our actively controlled switches, we expect to change the phase we are applying. The relationship between an electronic signal to the phase strength applied is dependent on the electro-optic modulator tecnology, and this relationship may also be nonlinear. Note that in practice, an analog mapping signal drives the phase response of the modulator which requires an analog circuit interconnect which might distort or drift the desired signal to apply. To start, we will explore an ideal digital-to-phase mapping and then extend this system modelling with analog circuit components and performance.
#
# ### Ideal Digital-to-Phase Mapping
#
# For example, assume we have a 4-bit DAC. We know that our applied phase shift $\phi=0$ at our digital code $b0000$. Assume we have an ideal linear phase-shifter that maps the code $b1111$ to $\phi=\pi$. `piel` provides a convenient function to extract this code-to-phase mapping:

basic_ideal_phase_map = piel.models.logic.electro_optic.linear_bit_phase_map(
    bits_amount=5, final_phase_rad=np.pi, initial_phase_rad=0
)
basic_ideal_phase_map

# |    |   bits |    phase |
# |---:|-------:|---------:|
# |  0 |      0 | 0        |
# |  1 |      1 | 0.101341 |
# |  2 |     10 | 0.202681 |
# |  3 |     11 | 0.304022 |
# |  4 |    100 | 0.405363 |
# |  5 |    101 | 0.506703 |
# |  6 |    110 | 0.608044 |
# |  7 |    111 | 0.709385 |
# |  8 |   1000 | 0.810726 |
# |  9 |   1001 | 0.912066 |
# | 10 |   1010 | 1.01341  |
# | 11 |   1011 | 1.11475  |
# | 12 |   1100 | 1.21609  |
# | 13 |   1101 | 1.31743  |
# | 14 |   1110 | 1.41877  |
# | 15 |   1111 | 1.52011  |
# | 16 |  10000 | 1.62145  |
# | 17 |  10001 | 1.72279  |
# | 18 |  10010 | 1.82413  |
# | 19 |  10011 | 1.92547  |
# | 20 |  10100 | 2.02681  |
# | 21 |  10101 | 2.12815  |
# | 22 |  10110 | 2.2295   |
# | 23 |  10111 | 2.33084  |
# | 24 |  11000 | 2.43218  |
# | 25 |  11001 | 2.53352  |
# | 26 |  11010 | 2.63486  |
# | 27 |  11011 | 2.7362   |
# | 28 |  11100 | 2.83754  |
# | 29 |  11101 | 2.93888  |
# | 30 |  11110 | 3.04022  |
# | 31 |  11111 | 3.14156  |
#

# This allows us to create an operational model of our phase shifter. It is also possible, that if we have a phase-voltage curve, we can also map that to the analog signal, and the analog signal to the DAC converter accordingly, when a Pandas dataframe is provided.
#
# ### Example simulation from our `cocotb` `simple_design` outputs
#
# We have some bit string simulation results from our `simple_design` `cocotb` simulation which is in the form of a simple Pandas dataframe as discussed in example `docs/examples/02_cocotb_simulation`
#

import simple_design

cocotb_simulation_output_files = piel.get_simulation_output_files_from_design(
    simple_design
)
example_simple_simulation_data = piel.read_simulation_data(
    cocotb_simulation_output_files[0]
)
example_simple_simulation_data

# |    |   Unnamed: 0 |    a |    b |     x |     t |
# |---:|-------------:|-----:|-----:|------:|------:|
# |  0 |            0 |  101 | 1010 |  1111 |  2001 |
# |  1 |            1 | 1001 | 1001 | 10010 |  4001 |
# |  2 |            2 |    0 | 1011 |  1011 |  6001 |
# |  3 |            3 |  100 |  101 |  1001 |  8001 |
# |  4 |            4 |  101 |    0 |   101 | 10001 |
# |  5 |            5 |   11 |    0 |    11 | 12001 |
# |  6 |            6 |  101 | 1011 | 10000 | 14001 |
# |  7 |            7 | 1000 |  101 |  1101 | 16001 |
# |  8 |            8 | 1101 |  100 | 10001 | 18001 |
# |  9 |            9 | 1001 |   11 |  1100 | 20001 |
# | 10 |           10 | 1011 | 1111 | 11010 | 22001 |

# We can get the phase that is mapped to this electronic data accordingly:

basic_ideal_phase_array = (
    piel.models.logic.electro_optic.return_phase_array_from_data_series(
        data_series=example_simple_simulation_data.x, phase_map=basic_ideal_phase_map
    )
)

# We can append this into our initial time-domain dataframe:

example_simple_simulation_data["phase"] = basic_ideal_phase_array
example_simple_simulation_data

# |    | Unnamed: 0 |   a   |   b   |   x   |   t   |  phase   |
# |---:|-----------:|------:|------:|------:|------:|---------:|
# |  0 |          0 |  101  | 1010  | 1111  | 2001  |  1.52011 |
# |  1 |          1 | 1001  | 1001  | 10010 | 4001  |  1.82413 |
# |  2 |          2 |   0   | 1011  | 1011  | 6001  |  1.11475 |
# |  3 |          3 |  100  |  101  | 1001  | 8001  | 0.912066 |
# |  4 |          4 |  101  |   0   |  101  | 10001 | 0.506703 |
# |  5 |          5 |  11   |   0   |  11   | 12001 | 0.304022 |
# |  6 |          6 |  101  | 1011  | 10000 | 14001 |  1.62145 |
# |  7 |          7 | 1000  |  101  | 1101  | 16001 |  1.31743 |
# |  8 |          8 | 1101  |  100  | 10001 | 18001 |  1.72279 |
# |  9 |          9 | 1001  |  11   | 1100  | 20001 |  1.21609 |
# | 10 |         10 | 1011  | 1111  | 11010 | 22001 |  2.63486 |

# This looks like this in GTKWave:

# ![example_simple_design_outputs](../_static/img/examples/02_cocotb_simulation/example_simple_design_outputs.PNG)

# ## Connecting into Active Unitary Calculations

# ### Simple Active 2x2 MZI Phase Shifter

# In order to determine the variation of the unitary dependent on an active phase, we need to first define our circuit model and which phase shifter we would be modulating. We will compose an active MZI2x2 switch based on the decomposition provided by the extracted `sax` netlist. First we determine what are our circuit missing models.

sax.get_required_circuit_models(mzi2x2_2x2_phase_shifter_netlist)

# ```['bend_euler', 'mmi2x2', 'straight', 'straight_heater_metal_simple']```

# We have some basic models in `piel` we can use to compose our circuit

all_models = piel.models.frequency.get_all_models()
all_models

straight_heater_metal_simple = all_models["ideal_active_waveguide"]
straight_heater_metal_simple

our_custom_library = piel.models.frequency.compose_custom_model_library_from_defaults(
    {"straight_heater_metal_simple": straight_heater_metal_simple}
)
our_custom_library

mzi2x2_model, mzi2x2_model_info = sax.circuit(
    netlist=mzi2x2_2x2_phase_shifter_netlist, models=our_custom_library
)
piel.sax_to_s_parameters_standard_matrix(mzi2x2_model(), input_ports_order=("o2", "o1"))

# ```python
# (Array([[-0.11039409-0.27826965j, -0.35184565-0.88689554j],
#         [-0.35184568-0.88689554j,  0.11039409+0.27826962j]],      dtype=complex64),
#  ('o2', 'o1'))
# ```

# Because we want to model the phase change applied from our heated waveguide, which we know previously corresponds to the `sxb` instance, we can recalculate our s-parameter matrix according to our applied phase:

piel.sax_to_s_parameters_standard_matrix(
    mzi2x2_model(sxt={"active_phase_rad": np.pi}),
    input_ports_order=(
        "o2",
        "o1",
    ),
)

# ```python
# (Array([[-0.88689834+0.35184222j,  0.2782662 -0.11039126j],
#         [ 0.2782662 -0.11039126j,  0.88689834-0.35184222j]],      dtype=complex64),
#  ('o2', 'o1'))
# ```

# We can clearly see our unitary is changing according to the `active_phase_rad` that we have applied to our circuit.

# #### Digital Data-Driven Active MZI 2x2

# Now we can compute what the unitary of our photonic circuit would be for each of the phases applied in our `cocotb` `simple_design` simulation outputs:

mzi2x2_active_unitary_array = list()
for phase_i in example_simple_simulation_data.phase:
    mzi2x2_active_unitary_i = piel.sax_to_s_parameters_standard_matrix(
        mzi2x2_model(sxt={"active_phase_rad": phase_i}),
        input_ports_order=(
            "o2",
            "o1",
        ),
    )
    mzi2x2_active_unitary_array.append(mzi2x2_active_unitary_i)

# We can copy this to a new dataframe and append the data in accordingly:

mzi2x2_simple_simulation_data = example_simple_simulation_data.copy()
mzi2x2_simple_simulation_data["unitary"] = mzi2x2_active_unitary_array
mzi2x2_simple_simulation_data

# Now we have a direct mapping between our digital state, time, and unitary changes in our `mzi` shifted circuit.

# #### Visualising Photonic and Electronic Data

# ##### Static

# Now we have computed how our photonic circuit changes based on an electronic input. Let us assume we are constantly inputting a power of 1dB on the `o2` top input port of the MZI, and we can measure the optical amplitude on the output.

optical_port_input = np.array([1, 0])
optical_port_input

# Let's run an example:

mzi2x2_simple_simulation_data.unitary.iloc[0]

example_optical_power_output = np.dot(
    mzi2x2_simple_simulation_data.unitary.iloc[0][0], optical_port_input
)
example_optical_power_output

# Now we can calculate this in our steady state in time:

output_amplitude_array_0 = np.array([])
output_amplitude_array_1 = np.array([])
for unitary_i in mzi2x2_simple_simulation_data.unitary:
    output_amplitude_i = np.dot(unitary_i[0], optical_port_input)
    output_amplitude_array_0 = np.append(
        output_amplitude_array_0, output_amplitude_i[0]
    )
    output_amplitude_array_1 = np.append(
        output_amplitude_array_1, output_amplitude_i[1]
    )
output_amplitude_array_0

# ```python
# array([-0.16433296+0.40858838j, -0.29093617+0.49160349j,
#        -0.04479861+0.24660602j, -0.01185814+0.15097517j,
#        -0.00628051-0.0502601j , -0.03387329-0.14756872j,
#        -0.20365313+0.4404957j , -0.09631109+0.3336508j ,
#        -0.24599898+0.46826234j, -0.06834814+0.29143432j,
#        -0.68521404+0.500675)`
#
#
#
# ```

mzi2x2_simple_simulation_data["output_amplitude_array_0"] = output_amplitude_array_0
mzi2x2_simple_simulation_data["output_amplitude_array_1"] = output_amplitude_array_1
mzi2x2_simple_simulation_data

# |    | Unnamed: 0 | a    | b    | x     | t     | phase    | unitary                                                                                                        | output_amplitude_array_0                      | output_amplitude_array_1                   |
# |---:|-----------:|-----:|-----:|------:|------:|---------:|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------|-------------------------------------------|
# |  0 | 0          |  101 | 1010 |  1111 |  2001 | 1.52011  | array([[-0.16426986+0.4086031j ,  0.33489325-0.83300986j],[ 0.33489325-0.83300986j,  0.16426986-0.4086031j ]]), ('o2', 'o1') | -0.16426986489554396+0.40860309522788557j  | 0.3348932484400226-0.8330098644113894j   |
# |  1 | 1          | 1001 | 1001 | 10010 |  4001 | 1.82413  | array([[-0.29089065+0.49165187j,  0.41794202-0.70638908j],[ 0.41794202-0.70638908j,  0.29089065-0.49165187j]]), ('o2', 'o1') | -0.2908906510099731+0.4916518717461718j    | 0.41794202495830884-0.7063890782969602j  |
# |  2 | 2          |    0 | 1011 |  1011 |  6001 | 1.11475  | array([[-0.04476771+0.24661686j,  0.17290701-0.95251202j],[ 0.17290701-0.95251202j,  0.04476771-0.24661686j]]), ('o2', 'o1') | -0.04476771225818654+0.24661685816779796j  | 0.172907011379935-0.9525120170487468j    |
# |  3 | 3          |  100 |  101 |  1001 |  8001 | 0.912066 | array([[-0.01183396+0.1509602j ,  0.07725035-0.98544577j],[ 0.07725035-0.98544577j,  0.01183396-0.1509602j ]]), ('o2', 'o1') | -0.01183396305024137+0.15096020006539462j  | 0.07725035327753166-0.985445766256692j   |
# |  4 | 4          |  101 | 0    | 101   | 10001 | 0.506703 | array([[-0.00628735-0.05025993j, -0.12396978-0.99099238j],[-0.12396978-0.99099238j,  0.00628735+0.05025993j]]), ('o2', 'o1') | -0.006287346214285949-0.050259929453309954j   | -0.12396977624117292-0.9909923830926474j    |
# |  5 | 5          |  11  | 0    | 11    | 12001 | 0.304022 | array([[-0.03390155-0.14758559j, -0.22129543-0.96337818j],[-0.22129543-0.96337818j,  0.03390155+0.14758559j]]), ('o2', 'o1') | -0.03390155326515837-0.14758558714522307j     | -0.22129543393308604-0.963378176041775j     |
# |  6 | 6          |  101 | 1011 | 10000 | 14001 | 1.62145  | array([[-0.20359414+0.44052313j, 0.36681329-0.79368558j],[ 0.36681329-0.79368558j,  0.20359414-0.44052313j]]), ('o2', 'o1')  | -0.2035941443096877+0.4405231323314459j       | 0.36681328554358295-0.7936855849972456j     |
# |  7 | 7          | 1000 | 101  | 1101  | 16001 | 1.31743  | array([[-0.09628268+0.33368601j, 0.25997616-0.90099705j],[ 0.25997616-0.90099705j,  0.09628268-0.33368601j]]), ('o2', 'o1')  | -0.0962826752595154+0.33368600691282485j      | 0.2599761601249619-0.9009970540474179j      |
# |  8 | 8          | 1101 | 100  | 10001 | 18001 | 1.72279  | array([[-0.24594593+0.46830107j, 0.39459122-0.7513338j ],[ 0.39459122-0.7513338j ,  0.24594593-0.46830107j]]), ('o2', 'o1') | -0.24594593237703394+0.46830106903747976j    | 0.3945912222496168-0.7513337969298994j      |
# |  9 | 9          | 1001 | 11   | 1100  | 20001 | 1.21609  | array([[-0.06831739+0.29145769j, 0.21774784-0.92896234j],[ 0.21774784-0.92896234j,  0.06831739+0.29145769j]]), ('o2', 'o1') | -0.0683173918484482+0.2914576912480423j      | 0.21774784446017936-0.9289623374584851j     |
# | 10 | 10         | 1011 | 1111 | 11010 | 22001 | 2.63486  | array([[-0.68513737+0.5007716j , 0.42706175-0.31214236j],[ 0.42706175-0.31214236j,  0.68513737-0.5007716j ]]), ('o2', 'o1') | -0.6851373654811859+0.5007715993013784j      | 0.42706175251351547-0.3121423638257475j     |

# This allows us to plot our optical signal amplitudes in the context of our active unitary variation, we can also simulate how optical inputs that are changing within the state of the unitary affect the total systems. However, for the sake of easy visualisation, we can begin to explore this. Note these results are just for trivial inputs.

# Note that we are trying to plot our signals amplitude, phase in time so it is a three dimensional visualisation.
#
# First, let's transform our complex data into amplitude and phase


mzi2x2_simple_simulation_data["output_amplitude_array_0_abs"] = np.abs(
    mzi2x2_simple_simulation_data.output_amplitude_array_0
)
mzi2x2_simple_simulation_data["output_amplitude_array_0_phase_rad"] = np.angle(
    mzi2x2_simple_simulation_data.output_amplitude_array_0
)
mzi2x2_simple_simulation_data["output_amplitude_array_0_phase_deg"] = np.angle(
    mzi2x2_simple_simulation_data.output_amplitude_array_0, deg=True
)
mzi2x2_simple_simulation_data["output_amplitude_array_1_abs"] = np.abs(
    mzi2x2_simple_simulation_data.output_amplitude_array_1
)
mzi2x2_simple_simulation_data["output_amplitude_array_1_phase_rad"] = np.angle(
    mzi2x2_simple_simulation_data.output_amplitude_array_1
)
mzi2x2_simple_simulation_data["output_amplitude_array_1_phase_deg"] = np.angle(
    mzi2x2_simple_simulation_data.output_amplitude_array_1, deg=True
)
mzi2x2_simple_simulation_data


# We will now convert the data into a plottable form, as when VCD or timing data files are parsed, they assume only a steady point and the plotter includes the lines. However, because we need to account for this type of co-simulation formats, we need to transform the data into a plotting form.

mzi2x2_simple_simulation_data_lines = piel.visual.points_to_lines_fixed_transient(
    data=mzi2x2_simple_simulation_data,
    time_index_name="t",
    fixed_transient_time=1,
)

# #### Basic Plots
#
# Here we are plotting how the electrical phase applied by the testbench 5-bit digital data, maps onto the optical phase applied on the heated waveguide, and we can use `sax` to measure the optical amplitude and phase at both ports of the MZI2x2.
#
# Note, that for now, we will assume that our applied optical phase is applied onto an ideal phase shifter, where the bandwidth is infinite, and where the applied operation translates to the optical input perfectly. We will make a more realistic time-dependent model of our circuit later.
#
# For the sake of simplicity, we can plot phase and amplitude over time driven by a digitally-encoded applied phase.

simple_ideal_o3_mzi_2x2_plots = piel.visual.plot_simple_multi_row(
    data=mzi2x2_simple_simulation_data_lines,
    x_axis_column_name="t",
    row_list=[
        "phase",
        "output_amplitude_array_0_abs",
        "output_amplitude_array_0_phase_deg",
    ],
    y_axis_title_list=["e1 Phase", "o3 Amplitude", "o3 Phase"],
)
simple_ideal_o3_mzi_2x2_plots.savefig(
    "../_static/img/examples/03a_sax_active_cosimulation/simple_ideal_o3_mzi_2x2_plots.PNG"
)

# ![simple_ideal_o3_mzi_2x2_plots](../_static/img/examples/03a_sax_active_cosimulation/simple_ideal_o3_mzi_2x2_plots.PNG)

simple_ideal_o4_mzi_2x2_plots = piel.visual.plot_simple_multi_row(
    data=mzi2x2_simple_simulation_data_lines,
    x_axis_column_name="t",
    row_list=[
        "phase",
        "output_amplitude_array_1_abs",
        "output_amplitude_array_1_phase_deg",
    ],
    y_axis_title_list=["e1 Phase", "o4 Amplitude", "o4 Phase"],
)
simple_ideal_o4_mzi_2x2_plots.savefig(
    "../_static/img/examples/03a_sax_active_cosimulation/simple_ideal_o4_mzi_2x2_plots.PNG"
)

# ![simple_ideal_o4_mzi_2x2_plots](../_static/img/examples/03a_sax_active_cosimulation/simple_ideal_o4_mzi_2x2_plots.PNG)

# ### Ideal Electronic Load Inaccuracy

# One thing we might want to do is to consider how the total electronic signals vary according to the electrical load that implements our phase shifter operation. In this case, it is a resistive heater. We might want to explore how the RC and heat dissipation dyanamics of our heater affects our full optical switching performance based on our digital input. This is further exemplified in the next example.


# ## Active MZI 2x2 Component Lattice

# Say we have a lattice with two different phase shifters this time. We want to see how the unitary changes when we apply phase control over a different set of parameters.

example_component_lattice = [
    [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2()],
    [0, mzi2x2_2x2(), 0],
    [mzi2x2_2x2(), 0, mzi2x2_2x2_phase_shifter()],
]

mixed_switch_lattice_circuit = gf.components.component_lattice_generic(
    network=example_component_lattice
)
# mixed_switch_circuit.show()
mixed_switch_lattice_circuit.plot_widget()

# ![switch_circuit_plot_widget](../_static/img/examples/03_sax_basics/switch_circuit_plot_widget.PNG)

# ### Model Composition

mixed_switch_lattice_circuit_netlist = (
    mixed_switch_lattice_circuit.get_netlist_recursive(
        exclude_port_types="electrical", allow_multiple=True
    )
)
mixed_switch_lattice_circuit_netlist.keys()

# ```python
# dict_keys(['component_lattice_gener_fb8c4da8', 'mzi_214beef3', 'straight_heater_metal_s_ad3c1693', 'via_stack_13a1ac5c', 'mzi_d46c281f'])
# ```
#
# This will exactly vary in your case.

mixed_switch_lattice_circuit_netlist["mzi_214beef3"]["instances"].keys()

# We can check what models we need to provide to compose the circuit. In our case, we want to determine all the instances that implement a particular model. This can be built directly into sax.

recursive_composed_required_models = sax.get_required_circuit_models(
    mixed_switch_lattice_circuit_netlist["component_lattice_gener_fb8c4da8"],
    models=piel.models.frequency.get_default_models(),
)
recursive_composed_required_models

# ```python
# ['mzi_214beef3', 'mzi_d46c281f']
# ```
#
# So this tells us all the models that are recursively composed, but not inherently provided by our defaults library. These are the models we can explore.

recursive_composed_required_models_0 = sax.get_required_circuit_models(
    mixed_switch_lattice_circuit_netlist[recursive_composed_required_models[0]],
    models=piel.models.frequency.get_default_models(),
)
recursive_composed_required_models_0

# ```python
# ['straight_heater_metal_s_ad3c1693']
# ```

piel.get_component_instances(
    mixed_switch_lattice_circuit_netlist,
    top_level_prefix="mzi_214beef3",
    component_name_prefix=recursive_composed_required_models_0[0],
)

# ```python
# {'straight_heater_metal_s_ad3c1693': ['sxt']}
# ```

sax.get_required_circuit_models(
    mixed_switch_lattice_circuit_netlist[recursive_composed_required_models[1]],
    models=piel.models.frequency.get_default_models(),
)

# ```python
# []
# ```

# Now, we know from our example above that we can go deeper down the rabbit hole of iterative models until we have provided all models for our device. Let's just look at this in practice:

recursive_composed_required_models_0_0 = sax.get_required_circuit_models(
    mixed_switch_lattice_circuit_netlist[recursive_composed_required_models_0[0]],
    models=piel.models.frequency.get_default_models(),
)
recursive_composed_required_models_0_0

# ```python
# []
# ```
#
# So this means that all the levels of the model can be composed from our default dictionary.

our_recursive_custom_library = (
    piel.models.frequency.compose_custom_model_library_from_defaults(
        {"straight_heater_metal_s_ad3c1693": straight_heater_metal_simple}
    )
)
our_recursive_custom_library

# What we can do now is that we can extract what instances use this model.

active_phase_shifters_dictionary = piel.get_component_instances(
    mixed_switch_lattice_circuit_netlist,
    top_level_prefix="component_lattice_gener_fb8c4da8",
    component_name_prefix=recursive_composed_required_models[0],
)
active_phase_shifters_dictionary

# ```python
# {'mzi_214beef3': ['mzi_1', 'mzi_5']}
# ```

# So these instances are our active phase shifters in our network.

# What `sax.netlist` does, is to map each instance with each component, and then `sax.circuit` maps each component with each model which is then multiplied together.

# ### Controlling our Phase Shifter Instances

# One major complexity we have is that we do not know where our phase shifters are. We can find them in the layout, but we need our algorithm to determine them. There are a few things we know about them for sure. We know that our phase shifter instances begin with `straight_heater_metal_s`. However, we do not yet algorithmically know where they are. We know we can do the following based on our previous analysis. So what we will do now is extract all the active phase shifter components, and their corresponding location within the netlist. Let's remember where we want to end:

(
    mixed_switch_lattice_circuit_s_parameters,
    mixed_switch_lattice_circuit_s_parameters_info,
) = sax.circuit(
    netlist=mixed_switch_lattice_circuit_netlist,
    models=our_recursive_custom_library,
)
piel.sax_to_s_parameters_standard_matrix(mixed_switch_lattice_circuit_s_parameters())
# mzi2x2_model(sxt={"active_phase_rad": phase_i}),

# ```python
# (Array([[ 0.23089845+0.23322447j, -0.13939448-0.2099313j ,
#           0.23096855+0.1446734j ,  0.5804817 -0.6461842j ],
#         [-0.03015644-0.250185j  , -0.92044485+0.087694j  ,
#          -0.05714459+0.06361263j,  0.16851723+0.21419339j],
#         [ 0.14126453+0.2330689j , -0.05715179+0.0636061j ,
#          -0.9265932 +0.09453729j, -0.04215275-0.2216345j ],
#         [ 0.58055454-0.6461182j ,  0.2467988 +0.1156164j ,
#          -0.13727726-0.17903534j,  0.3338281 +0.09417956j]],      dtype=complex64),
#  ('in_o_0', 'in_o_1', 'in_o_2', 'in_o_3'))
# ```

# + active=""
# mixed_switch_lattice_circuit_s_parameters
# -

active_phase_shifters_dictionary[recursive_composed_required_models[0]][0]

piel.sax_to_s_parameters_standard_matrix(
    mixed_switch_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": np.pi}}
    )
)

# ```python
# (Array([[-0.07260128-0.2413117j , -0.8917833 +0.2441996j ,
#          -0.04539197+0.07246678j,  0.20274116+0.1821285j ],
#         [ 0.18749169+0.26935905j, -0.10133702-0.23071809j,
#           0.20274433+0.18213132j,  0.6826765 -0.5370923j ],
#         [ 0.14126453+0.2330689j , -0.05715179+0.0636061j ,
#          -0.9265932 +0.09453729j, -0.04215275-0.2216345j ],
#         [ 0.58055454-0.6461182j ,  0.2467988 +0.1156164j ,
#          -0.13727726-0.17903534j,  0.3338281 +0.09417956j]],      dtype=complex64),
#  ('in_o_0', 'in_o_1', 'in_o_2', 'in_o_3'))
# ```

# You can clearly see the position and change of the s-parameter matrix accordingly.

piel.sax_to_s_parameters_standard_matrix(
    mixed_switch_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": np.pi}},
        mzi_5={"sxt": {"active_phase_rad": np.pi}},
    )
)


# ```python
# (Array([[-0.07260128-0.2413117j , -0.8917833 +0.2441996j ,
#           0.23096433+0.14467366j, -0.05714355+0.06361032j],
#         [ 0.18749169+0.26935905j, -0.10133702-0.23071809j,
#           0.58048916-0.646181j  ,  0.16851316+0.21419221j],
#         [ 0.14126453+0.2330689j , -0.05715179+0.0636061j ,
#          -0.07952578-0.21112217j, -0.92908216-0.06572803j],
#         [ 0.58055454-0.6461182j ,  0.2467988 +0.1156164j ,
#           0.34503105+0.03555341j, -0.10454827-0.19992213j]],      dtype=complex64),
#  ('in_o_0', 'in_o_1', 'in_o_2', 'in_o_3'))
# ```

# However, we want to control the phase shifting effect and control the component we are modifying. In our case, we want to modify the phase of the model controlled by our thermo-optic phase shifters which are `straight_heater_metal_s_*` instances. So let's find all the instances and corresponding models where there is one of these models. We know from the `required_models` function that we are have distinct models required for each of these Mach-Zeneder components.

# From this we can tell only of the corresponding instances and submodels. It is important to note that some of the models can be composed from other models, which means that you need to explore the composition of the internal components potentially if you want to do a full circuit composition verification. What we need to do now, is extract a list of our phase shifters that we can then apply our phase to.

# ### Algorithmically Extracted Phase-Shifter Instances

# So in this example, we're controlling the `sxt: {"active_phase_rad": ourphase}` but this is composed of the `ideal_active_waveguide` model and the corresponding `active_phase_rad` parameter but this is determined from the `straight_heater_metal_s_ad3c1693` definition of the `straight_heater_metal_simple` phase shifter component. You can see the complexity of the system construction. We can extract all instances that contain this component which are `mzi_1` and `mzi_5`.
#
# However, we want to create a map that returns a phase list in this form:
#
# \begin{equation}
# \left[ \phi \right] = \left[ \phi_0, \phi_1, \phi_2 ... \phi_N \right]
# \end{equation}
#
# However, then we need to determine the index. Let's determine the


def compose_recursive_instance_location(
    recursive_netlist: dict,
    top_level_instance_name: str,
    required_models: list,
    target_component_prefix: str,
    models: dict,
):
    """
       This function returns the recursive location of any matching ``target_component_prefix`` instances within the ``recursive_netlist``. A function that returns the mapping of the ``matched_component`` in the corresponding netlist at any particular level of recursion. This function iterates over a particular level of recursion of a netlist. It returns a list of the missing required components, and updates a dictionary of models that contains a particular matching component. It returns the corresponding list of instances of a particular component at that level of recursion, so that it can be appended upon in order to construct the location of the corresponding matching elements.

       If ``required_models`` is an empty list, it means no recursion is required and the function is complete. If a ``required_model_i`` in ``required_models`` matches ``target_component_prefix``, then no more recursion is required down the component function.

       The ``recursive_netlist`` should contain all the missing composed models that are not provided in the main models dictionary. If not, then we need to require the user to input the missing model that cannot be extracted from the composed netlist.
    We know when a model is composed, and when it is already provided at every level of recursion based on the ``models`` dictionary that gets updated at each level of recursion with the corresponding models of that level, and the ``required_models`` down itself.

       However, a main question appears on how to do the recursion. There needs to be a flag that determines that the recursion is complete. However, this is only valid for every particular component in the ``required_models`` list. Every component might have missing component. This means that this recursion begins component by component, updating the ``required_models`` list until all of them have been composed from the recursion or it is determined that is it missing fully.

       It would be ideal to access the particular component that needs to be implemented.

       Returns a tuple of ``model_composition_mapping, instance_composition_mapping, target_component_mapping`` in the form of

           ({'mzi_214beef3': ['straight_heater_metal_s_ad3c1693']},
            {'mzi_214beef3': ['mzi_1', 'mzi_5'],
             'mzi_d46c281f': ['mzi_2', 'mzi_3', 'mzi_4']})
    """
    model_composition_mapping = dict()
    instance_composition_mapping = dict()
    target_component_mapping = dict()
    i = 0
    while len(required_models) != 0:
        # if len(required_models) == 0:
        #     pass
        #     # Return the results as the recursive iteration is now complete.
        # else:
        # TODO Break if required_models cannot be composed and needs to be provided by the user.
        # This means that the model inside the top_level required model also has a required model that should be inside the recursive netlist and we need to find it.
        # We iterate over each of the required model names to see if they match our active component name.
        for required_model_name_i in required_models:
            # Appends required_models_i from subcomponent to the required_models input based on the models provided.
            required_models_i = sax.get_required_circuit_models(
                recursive_netlist[
                    required_model_name_i
                ],  # TODO make this recursive so it can search inside? This will never have to be 2D as all models outside.
                models={**models, **model_composition_mapping},
            )  # eg. ["straight_heater_metal_s_ad3c1693"]

            # Check if required_model_name_i already composed.

            # Check that the model composition mapping has not already fulfilled this model.
            if len(required_models_i) != 0:
                if required_model_name_i in model_composition_mapping:
                    required_models.remove(required_model_name_i)
                else:
                    required_models.extend(required_models_i)
                    model_composition_mapping[required_model_name_i] = required_models_i
            elif len(required_models_i) == 0:
                # Remove from ``required_models`` to complete the recursion
                required_models.remove(required_model_name_i)

            # Get the corresponding instances of this model at this level of recursion.
            # Implement a function that matches all the potential corresponding matched instances on the top_level
            instance_composition_mapping_i = piel.get_component_instances(
                recursive_netlist=recursive_netlist,
                top_level_prefix=top_level_instance_name,
                component_name_prefix=required_model_name_i,
            )  # {'mzi_214beef3': ['mzi_1', 'mzi_5']}
            instance_composition_mapping.update(instance_composition_mapping_i)

            # This model is now at a particular level of recursion, let's check if this is the model we want in the required composed models.
            for required_model_name_i_i in required_models_i:
                if required_model_name_i_i.startswith(target_component_prefix):
                    # Yes, this is the model we want. Can we compose the instance location?
                    target_component_mapping.update(
                        {required_model_name_i_i: required_model_name_i}
                    )
                    # This means we need to check whether the components is our matched component, and if not, then we need to check if this other required component recursively also requires our active component. Implement the search again recursively from the unmatched components.

            # If the target_component has the corresponding mapping in the recursive_netlist then we can access the lowest component element
            if required_model_name_i.startswith(target_component_prefix):
                if required_model_name_i in target_component_mapping:
                    instance_composition_mapping_i = piel.get_component_instances(
                        recursive_netlist=recursive_netlist,
                        top_level_prefix=target_component_mapping[
                            required_model_name_i
                        ],
                        component_name_prefix=required_model_name_i,
                    )  # {'mzi_214beef3': ['mzi_1', 'mzi_5']}
                    instance_composition_mapping.update(instance_composition_mapping_i)

        i += 1

    return (
        model_composition_mapping,
        instance_composition_mapping,
        target_component_mapping,
    )


from typing import Optional


def get_matched_model_recursive_netlist_instances(
    recursive_netlist: dict,
    top_level_instance_prefix: str,
    target_component_prefix: str,
    models: Optional[dict] = None,
) -> list[tuple]:
    """
    This function returns an active component list with a tuple mapping of the location of the active component within the recursive netlist and corresponding model. It will recursively look within a netlist to locate what models use a particular component model. At each stage of recursion, it will compose a list of the elements that implement this matching model in order to relate the model to the instance, and hence the netlist address of the component that needs to be updated in order to functionally implement the model.

    It takes in as a set of parameters the recursive_netlist generated by a ``gdsfactory`` netlist implementation.

    Returns a list of tuples, that correspond to the phases applied with the corresponding component paths at multiple levels of recursion.
    eg. [("component_lattice_gener_fb8c4da8", "mzi_1", "sxt"), ("component_lattice_gener_fb8c4da8", "mzi_5", "sxt")] and these are our keys to our sax circuit decomposition.
    """
    matched_instance_list = []
    if models is None:
        models = piel.models.frequency.get_default_models()

    # We need to input the top-level instance.
    top_level_instance_name = piel.get_netlist_instances_by_prefix(
        recursive_netlist=mixed_switch_lattice_circuit_netlist,
        instance_prefix=top_level_instance_prefix,
    )

    # We need to input the prefix of the component of the straight metal heater.
    top_level_required_models = sax.get_required_circuit_models(
        recursive_netlist[top_level_instance_name],
        models=models,
    )

    (
        model_composition_mapping,
        instance_composition_mapping,
        target_component_mapping,
    ) = compose_recursive_instance_location(
        recursive_netlist=recursive_netlist,
        top_level_instance_name=top_level_instance_name,
        required_models=top_level_required_models.copy(),
        target_component_prefix=target_component_prefix,
        models=models,
    )

    # Now we have the raw data that creates the mapping of the components-to-instances, in order to create the corresponding instance address indexes that we can use to control our matching element parameters.
    if len(target_component_mapping.keys()) != 0:
        # This means that the target_component has been mapped to a parent recursive_netlist cell.
        for target_component_name_i in target_component_mapping.keys():
            # Tells us the name of our component
            recursive_parent_component_i = target_component_mapping[
                target_component_name_i
            ]  # Get the parent cell.
            for parent_instances_i in instance_composition_mapping[
                recursive_parent_component_i
            ]:
                # TODO check parent_instances_i not in target_component_mapping to increase hierarchy.
                # TODO implement as another recursive problem. NMP right now.
                for target_instances_i in instance_composition_mapping[
                    target_component_name_i
                ]:
                    matched_instance_list.append(
                        (
                            top_level_instance_name,
                            parent_instances_i,
                            target_instances_i,
                        )
                    )
    return matched_instance_list


get_matched_model_recursive_netlist_instances(
    recursive_netlist=mixed_switch_lattice_circuit_netlist,
    top_level_instance_prefix="component_lattice_gener",
    target_component_prefix="straight_heater_metal_s",
)


def b():
    # Get a netlist, and provide our models accordingly.
    (
        mixed_switch_lattice_circuit_s_parameters,
        mixed_switch_lattice_circuit_s_parameters_info,
    ) = sax.circuit(
        netlist=mixed_switch_lattice_circuit_netlist,
        models=our_recursive_custom_library,
    )
    # Create a tuple of the corresponding phase shifter positions we can input into other functions.

    # Return the phase shifter controller accordingly
    # Find a way to transform this information into corresponding phases, or maybe map a set of inputs accordingly.
    mixed_switch_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": np.pi}},
        mzi_5={"sxt": {"active_phase_rad": np.pi}},
    )
