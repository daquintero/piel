# # SAX Active Co-simulation

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2
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

# + active=""
# mixed_switch_circuit_netlist["connections"]
