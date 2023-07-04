# # SAX Active Co-simulation

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2
import numpy as np
import piel  # NOQA : F401
import sax  # NOQA : F401

# ## Active Generic Controlled Lattice

# First, let's look at our actively driven component

mzi2x2_2x2_phase_shifter_netlist = mzi2x2_2x2_phase_shifter().get_netlist()
mzi2x2_2x2_phase_shifter_netlist["instances"].keys()

# Note that the netlist is very identical to a non-actively driven component in a recursive implementation.

example_component_lattice = [
    [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=80.0)],
    [0, mzi2x2_2x2(delta_length=50.0), 0],
    [mzi2x2_2x2(delta_length=100.0), 0, mzi2x2_2x2_phase_shifter()],
]

mixed_switch_circuit = gf.components.component_lattice_generic(
    network=example_component_lattice
)
mixed_switch_circuit.plot_widget()

# ![img](../_static/img/examples/03_sax_basics/switch_circuit_plot_widget.PNG)

mixed_switch_circuit_netlist = mixed_switch_circuit.get_netlist(
    exclude_port_types="electrical"
)
mixed_switch_circuit_netlist["instances"].keys()

# ```
# dict_keys(['bend_euler_1', 'bend_euler_10', 'bend_euler_11', 'bend_euler_12', 'bend_euler_13', 'bend_euler_14', 'bend_euler_15', 'bend_euler_16', 'bend_euler_17', 'bend_euler_18', 'bend_euler_19', 'bend_euler_2', 'bend_euler_20', 'bend_euler_21', 'bend_euler_22', 'bend_euler_23', 'bend_euler_24', 'bend_euler_25', 'bend_euler_26', 'bend_euler_27', 'bend_euler_28', 'bend_euler_29', 'bend_euler_3', 'bend_euler_30', 'bend_euler_31', 'bend_euler_32', 'bend_euler_33', 'bend_euler_34', 'bend_euler_35', 'bend_euler_36', 'bend_euler_37', 'bend_euler_38', 'bend_euler_39', 'bend_euler_4', 'bend_euler_40', 'bend_euler_5', 'bend_euler_6', 'bend_euler_7', 'bend_euler_8', 'bend_euler_9', 'mzi_1', 'mzi_2', 'mzi_3', 'mzi_4', 'mzi_5', 'straight_1', 'straight_10', 'straight_11', 'straight_12', 'straight_13', 'straight_14', 'straight_15', 'straight_16', 'straight_17', 'straight_18', 'straight_19', 'straight_2', 'straight_20', 'straight_21', 'straight_22', 'straight_23', 'straight_24', 'straight_25', 'straight_26', 'straight_27', 'straight_28', 'straight_29', 'straight_3', 'straight_30', 'straight_31', 'straight_32', 'straight_33', 'straight_34', 'straight_35', 'straight_36', 'straight_37', 'straight_38', 'straight_39', 'straight_4', 'straight_40', 'straight_41', 'straight_42', 'straight_43', 'straight_44', 'straight_45', 'straight_46', 'straight_47', 'straight_48', 'straight_49', 'straight_5', 'straight_50', 'straight_51', 'straight_52', 'straight_53', 'straight_54', 'straight_55', 'straight_56', 'straight_57', 'straight_58', 'straight_59', 'straight_6', 'straight_60', 'straight_61', 'straight_62', 'straight_63', 'straight_64', 'straight_65', 'straight_66', 'straight_67', 'straight_68', 'straight_69', 'straight_7', 'straight_70', 'straight_71', 'straight_72', 'straight_73', 'straight_74', 'straight_75', 'straight_76', 'straight_77', 'straight_78', 'straight_8', 'straight_9'])
# ```

mixed_switch_circuit_netlist["ports"].keys()

# To extract the connectivity data

mixed_switch_circuit_netlist["connections"]

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
# ## Example Electro-Optic Operational Connectivity
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
