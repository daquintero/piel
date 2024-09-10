# # SAX Integration Basics

# In this example, we will explore different methodologies of mapping electronic signals to photonic operations. We will start by an ideal basic example, and explore the complexity of how these systems can be interconnected accordingly. We will explore different encodings of transformations between electronic simulation implementations and corresponding photonic solutions.

# In order to solve a photonic circuit using `sax`, we first need a physical netlist of our circuit that represents the inputs and outputs that we care about of our circuit.

# ## Basic Generic Component Lattice
#
# I actually designed this component so let me know what you would like to find in it!

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2
import piel
import sax

from gdsfactory.generic_tech import get_generic_pdk

PDK = get_generic_pdk()
PDK.activate()


# +
# dir(gf.components)
# -

# We create a balanced MZI lattice full of the same `mzi2x2_2x2` components to demonstrate `sax` network basics.

# +
balanced_mzi_lattice = [
    [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    [0, mzi2x2_2x2(), 0],
    [mzi2x2_2x2(), 0, mzi2x2_2x2()],
]

switch_circuit = gf.components.component_lattice_generic(network=balanced_mzi_lattice)
switch_circuit
# -

# ![img](../_static/img/examples/03_sax_basics/default_switch_circuit_plot_widget.PNG)

# ### Extract Netlist

# There are several ways to extract and model a photonic circuit such as this one using `gdsfactory`

top_level_netlist = switch_circuit.get_netlist()
# top_level_netlist

# ```python
# dict_keys(['bend_euler_1', 'bend_euler_2', 'bend_euler_3', 'bend_euler_4', 'bend_euler_5', 'bend_euler_6', 'bend_euler_7', 'bend_euler_8', 'cp1', 'cp2', 'straight_10', 'straight_5', 'straight_6', 'straight_7', 'straight_8', 'straight_9', 'sxb', 'sxt', 'syl', 'sytl'])
# ```

# We extract the top level circuit netlists accordingly, and see what instances are available:

top_level_netlist["instances"].keys()

# ```python
# dict_keys(['bend_euler_1', 'bend_euler_10', 'bend_euler_11', 'bend_euler_12', 'bend_euler_13', 'bend_euler_14', 'bend_euler_15', 'bend_euler_16', 'bend_euler_17', 'bend_euler_18', 'bend_euler_19', 'bend_euler_2', 'bend_euler_20', 'bend_euler_21', 'bend_euler_22', 'bend_euler_23', 'bend_euler_24', 'bend_euler_25', 'bend_euler_26', 'bend_euler_27', 'bend_euler_28', 'bend_euler_29', 'bend_euler_3', 'bend_euler_30', 'bend_euler_31', 'bend_euler_32', 'bend_euler_33', 'bend_euler_34', 'bend_euler_35', 'bend_euler_36', 'bend_euler_37', 'bend_euler_38', 'bend_euler_39', 'bend_euler_4', 'bend_euler_40', 'bend_euler_5', 'bend_euler_6', 'bend_euler_7', 'bend_euler_8', 'bend_euler_9', 'mzi_1', 'mzi_2', 'mzi_3', 'mzi_4', 'mzi_5', 'straight_1', 'straight_10', 'straight_11', 'straight_12', 'straight_13', 'straight_14', 'straight_15', 'straight_16', 'straight_17', 'straight_18', 'straight_19', 'straight_2', 'straight_20', 'straight_21', 'straight_22', 'straight_23', 'straight_24', 'straight_25', 'straight_26', 'straight_27', 'straight_28', 'straight_29', 'straight_3', 'straight_30', 'straight_31', 'straight_32', 'straight_33', 'straight_34', 'straight_35', 'straight_36', 'straight_37', 'straight_38', 'straight_39', 'straight_4', 'straight_40', 'straight_41', 'straight_42', 'straight_43', 'straight_44', 'straight_45', 'straight_46', 'straight_47', 'straight_48', 'straight_49', 'straight_5', 'straight_50', 'straight_51', 'straight_52', 'straight_53', 'straight_54', 'straight_55', 'straight_56', 'straight_57', 'straight_58', 'straight_59', 'straight_6', 'straight_60', 'straight_61', 'straight_62', 'straight_63', 'straight_64', 'straight_65', 'straight_66', 'straight_67', 'straight_68', 'straight_69', 'straight_7', 'straight_70', 'straight_71', 'straight_72', 'straight_73', 'straight_74', 'straight_75', 'straight_76', 'straight_77', 'straight_78', 'straight_8', 'straight_9'])
# ```

# Likewise the ports, note that there is a clear definition of the input and output ports numbered based on the ports top-down row endian.

top_level_netlist["ports"].keys()

# This is equivalent to extract a netlist recursively to the lower element levels on our component, just that with the recursive netlist we have a multi-component netlist list down to the deepest level, and allows us to model very thoroughly our circuit. It is this

recursive_netlist = switch_circuit.get_netlist_recursive()
# recursive_netlist

recursive_netlist[list(recursive_netlist.keys())[0]]["instances"].keys()

# We can also extract a flat netlist

flat_netlist_keys = switch_circuit.get_netlist_flat()["instances"].keys()

# Understanding the photonic netlisting functions are important when actively mapping our component measurement to these functions.

# ### Individual Components

# We could do the same thing for our individual components. For now, we only care about the optical ports to model the frequency networking.

mzi2x2_netlist = mzi2x2_2x2().get_netlist(exclude_port_types="electrical")
mzi2x2_netlist["instances"].keys()

# You can get an idea of the component ports

mzi2x2_netlist["ports"].keys()

# ```python
# dict_keys(['o1', 'o2', 'o4', 'o3'])
# ```

# Worth noting that in this model it is defined by passive measurement, whereas ours above has composite components, and this is dependent on the netlisting function and implementation.
#
# We have contributed to `sax` in order to `piel` provides some functionality to extract the base measurement that we require to define:

sax.get_required_circuit_models(mzi2x2_netlist)

# ```python
# ['bend_euler', 'mmi2x2', 'straight']
# ```

# #### Basic Models

# We will now include a basic list of measurement to do a `sax` s-parameter network simulation of the circuit. Based on the [layout-aware sax example](https://flaport.github.io/sax/examples/07_layout_aware.html), the `measurement` are a dictionary that maps the instance name to a functional `s-parameter` representation of the circuit.
#
# `piel` conveniently provides a library of component measurement that can be used to co-simulate electronic and photonic systems.

# Now we need to include our device measurement, we will start with basic ones and expand from that. It is easy to create some measurement as the components are all the in this default example.


# `piel` provides a library with a list of measurement, that we hope we can extend and improve with your contribution! We create our model dictionary accordingly based on our default photonic frequency library:

piel.models.frequency.get_default_models()

# Let's explore one of our default measurement. Each model has its source in the documentation.

piel.models.frequency.get_default_models()["straight"]()

sax_netlist = sax.netlist(mzi2x2_netlist)
sax.get_component_instances(
    recursive_netlist=sax_netlist,
    top_level_prefix="",
    component_name_prefix="bend_euler",
)

mzi2x2_model, mzi2x2_model_info = sax.circuit(
    netlist=mzi2x2_netlist, models=piel.models.frequency.get_default_models()
)
mzi2x2_model

# We can easily evaluate our model:

mzi2x2_model()

# #### Verify Instance Model Integration

# Let us explore how this function works, in order to understand, how our circuit model can be integrated into an actively quasi-static solver. Let us change the function of one instance in particular.

mzi2x2_model(bend_euler_1={"length": 15})

# ```python
# {('o1', 'o1'): Array(0.+0.j, dtype=complex128),
#  ('o1', 'o2'): Array(0.+0.j, dtype=complex128),
#  ('o1', 'o4'): Array(-0.38457016-0.81126986j, dtype=complex128),
#  ('o1', 'o3'): Array(-0.18864067-0.39794686j, dtype=complex128),
#  ('o2', 'o1'): Array(0.+0.j, dtype=complex128),
#  ('o2', 'o2'): Array(0.+0.j, dtype=complex128),
#  ('o2', 'o4'): Array(-0.18864067-0.39794686j, dtype=complex128),
#  ('o2', 'o3'): Array(0.38457016+0.81126986j, dtype=complex128),
#  ('o4', 'o1'): Array(-0.38457016-0.81126986j, dtype=complex128),
#  ('o4', 'o2'): Array(-0.18864067-0.39794686j, dtype=complex128),
#  ('o4', 'o4'): Array(0.+0.j, dtype=complex128),
#  ('o4', 'o3'): Array(0.+0.j, dtype=complex128),
#  ('o3', 'o1'): Array(-0.18864067-0.39794686j, dtype=complex128),
#  ('o3', 'o2'): Array(0.38457016+0.81126986j, dtype=complex128),
#  ('o3', 'o4'): Array(0.+0.j, dtype=complex128),
#  ('o3', 'o3'): Array(0.+0.j, dtype=complex128)}
# ```

# Seeing the evaluation outputs it is easy to note how the unitary changes based on dependent inputs.

mzi2x2_model(bend_euler_1={"length": 15}, bend_euler_3={"length": 20})

# ```python
# {('o1', 'o1'): Array(0.+0.j, dtype=complex128),
#  ('o1', 'o2'): Array(0.+0.j, dtype=complex128),
#  ('o1', 'o4'): Array(-0.1365982-0.97898445j, dtype=complex128),
#  ('o1', 'o3'): Array(-0.02092607-0.1499749j, dtype=complex128),
#  ('o2', 'o1'): Array(0.+0.j, dtype=complex128),
#  ('o2', 'o2'): Array(0.+0.j, dtype=complex128),
#  ('o2', 'o4'): Array(-0.02092607-0.1499749j, dtype=complex128),
#  ('o2', 'o3'): Array(0.1365982+0.97898445j, dtype=complex128),
#  ('o4', 'o1'): Array(-0.1365982-0.97898445j, dtype=complex128),
#  ('o4', 'o2'): Array(-0.02092607-0.1499749j, dtype=complex128),
#  ('o4', 'o4'): Array(0.+0.j, dtype=complex128),
#  ('o4', 'o3'): Array(0.+0.j, dtype=complex128),
#  ('o3', 'o1'): Array(-0.02092607-0.1499749j, dtype=complex128),
#  ('o3', 'o2'): Array(0.1365982+0.97898445j, dtype=complex128),
#  ('o3', 'o4'): Array(0.+0.j, dtype=complex128),
#  ('o3', 'o3'): Array(0.+0.j, dtype=complex128)}
# ```

# This is the power of `sax`, that it allows us to create large-scale interconnectivity and perform very fast evaluation.

# ## Passive Network Analysis

# Using these basic measurement we can also extract the total network performance of our composed generic_network_lattice:

# We can extract the whole netlist using a recursive netlist extraction:

switch_circuit_model, switch_circuit_model_info = sax.circuit(
    netlist=recursive_netlist,
    models=piel.models.frequency.get_default_models(),
)

# We can get the unitary accordingly:

switch_circuit_model()

# ```python
# {('in_o_0', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'out_o_0'): Array(0.24680707-0.15921314j, dtype=complex128),
#  ('in_o_0', 'out_o_1'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('in_o_0', 'out_o_2'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('in_o_0', 'out_o_3'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('in_o_1', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'out_o_0'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('in_o_1', 'out_o_1'): Array(-0.17334637+0.8963379j, dtype=complex128),
#  ('in_o_1', 'out_o_2'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('in_o_1', 'out_o_3'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('in_o_2', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'out_o_0'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('in_o_2', 'out_o_1'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('in_o_2', 'out_o_2'): Array(-0.17250486+0.90549771j, dtype=complex128),
#  ('in_o_2', 'out_o_3'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('in_o_3', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'out_o_0'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('in_o_3', 'out_o_1'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('in_o_3', 'out_o_2'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('in_o_3', 'out_o_3'): Array(0.31510862-0.05641407j, dtype=complex128),
#  ('out_o_0', 'in_o_0'): Array(0.24680707-0.15921314j, dtype=complex128),
#  ('out_o_0', 'in_o_1'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('out_o_0', 'in_o_2'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('out_o_0', 'in_o_3'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('out_o_0', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'in_o_0'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('out_o_1', 'in_o_1'): Array(-0.17334637+0.8963379j, dtype=complex128),
#  ('out_o_1', 'in_o_2'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('out_o_1', 'in_o_3'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('out_o_1', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'in_o_0'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('out_o_2', 'in_o_1'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('out_o_2', 'in_o_2'): Array(-0.17250486+0.90549771j, dtype=complex128),
#  ('out_o_2', 'in_o_3'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('out_o_2', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'in_o_0'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('out_o_3', 'in_o_1'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('out_o_3', 'in_o_2'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('out_o_3', 'in_o_3'): Array(0.31510862-0.05641407j, dtype=complex128),
#  ('out_o_3', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_3'): Array(0.+0.j, dtype=complex128)}
# ```

# However, we have a fundamental limitation here, this circuit is for a passive network. It calculates the phase and corresponding effects from the waveguide elements, but it does not include an active phase. So we need to think about ways to apply this. We can do this in a particular way as follows.

# ## Phase-Dependent Network Analysis

# Note that a recursive netlist allows us to construct our unitary based on fundamental elements, we can see the measurement it requires:

sax.get_required_circuit_models(recursive_netlist)

# We can also work out our existing standard netlist.

sax.get_required_circuit_models(top_level_netlist)

# Note the differences in between the recursive and standard netlist: the recursive netlist determines the `mzi_someinstance` from its subcomponents, whilst the standard model expects us to provide this model. For example, we could create a `sax` circuit using the `mzi2x2_2x2` recursive extracted model we composed previously. As such, we are doing `sax` multi-model-function composition.

new_models_library = piel.models.frequency.compose_custom_model_library_from_defaults(
    {"mzi": mzi2x2_model}
)
new_models_library

# Now we can create a circuit model from this composed function:

active_switch_circuit_model, active_switch_circuit_model_info = sax.circuit(
    netlist=top_level_netlist, models=new_models_library
)
active_switch_circuit_model()

# ```python
# {('in_o_0', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_0', 'out_o_0'): Array(0.24680707-0.15921314j, dtype=complex128),
#  ('in_o_0', 'out_o_1'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('in_o_0', 'out_o_2'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('in_o_0', 'out_o_3'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('in_o_1', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_1', 'out_o_0'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('in_o_1', 'out_o_1'): Array(-0.17334637+0.8963379j, dtype=complex128),
#  ('in_o_1', 'out_o_2'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('in_o_1', 'out_o_3'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('in_o_2', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_2', 'out_o_0'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('in_o_2', 'out_o_1'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('in_o_2', 'out_o_2'): Array(-0.17250486+0.90549771j, dtype=complex128),
#  ('in_o_2', 'out_o_3'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('in_o_3', 'in_o_0'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_1'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_2'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'in_o_3'): Array(0.+0.j, dtype=complex128),
#  ('in_o_3', 'out_o_0'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('in_o_3', 'out_o_1'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('in_o_3', 'out_o_2'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('in_o_3', 'out_o_3'): Array(0.31510862-0.05641407j, dtype=complex128),
#  ('out_o_0', 'in_o_0'): Array(0.24680707-0.15921314j, dtype=complex128),
#  ('out_o_0', 'in_o_1'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('out_o_0', 'in_o_2'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('out_o_0', 'in_o_3'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('out_o_0', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_0', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'in_o_0'): Array(-0.27779553+0.08816013j, dtype=complex128),
#  ('out_o_1', 'in_o_1'): Array(-0.17334637+0.8963379j, dtype=complex128),
#  ('out_o_1', 'in_o_2'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('out_o_1', 'in_o_3'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('out_o_1', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_1', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'in_o_0'): Array(0.26310001-0.07108837j, dtype=complex128),
#  ('out_o_2', 'in_o_1'): Array(0.00782271+0.08514974j, dtype=complex128),
#  ('out_o_2', 'in_o_2'): Array(-0.17250486+0.90549771j, dtype=complex128),
#  ('out_o_2', 'in_o_3'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('out_o_2', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_2', 'out_o_3'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'in_o_0'): Array(-0.07946653-0.86498831j, dtype=complex128),
#  ('out_o_3', 'in_o_1'): Array(0.27164846+0.02196099j, dtype=complex128),
#  ('out_o_3', 'in_o_2'): Array(-0.26202477-0.00849415j, dtype=complex128),
#  ('out_o_3', 'in_o_3'): Array(0.31510862-0.05641407j, dtype=complex128),
#  ('out_o_3', 'out_o_0'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_1'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_2'): Array(0.+0.j, dtype=complex128),
#  ('out_o_3', 'out_o_3'): Array(0.+0.j, dtype=complex128)}
# ```

# This gives us a lot of power, and which is why I love `sax`. What we can do now is modify the composed `sax` function to add a simple active phase parameter. We will start simple and extend this more thoroguhly to different electronic signals representations in the next example. We can represent this matrix in standard S-parameter matrix form using `piel`:

piel.sax_to_s_parameters_standard_matrix(active_switch_circuit_model())

# ```python
# (Array([[ 0.24680707-0.15921314j, -0.27779553+0.08816013j,
#           0.26310001-0.07108837j, -0.07946653-0.86498831j],
#         [-0.27779553+0.08816013j, -0.17334637+0.8963379j ,
#           0.00782271+0.08514974j,  0.27164846+0.02196099j],
#         [ 0.26310001-0.07108837j,  0.00782271+0.08514974j,
#          -0.17250486+0.90549771j, -0.26202477-0.00849415j],
#         [-0.07946653-0.86498831j,  0.27164846+0.02196099j,
#          -0.26202477-0.00849415j,  0.31510862-0.05641407j]],      dtype=complex128),
#  ('in_o_0', 'in_o_1', 'in_o_2', 'in_o_3'))
#
# ```
