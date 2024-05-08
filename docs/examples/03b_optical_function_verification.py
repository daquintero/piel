# # Optical Function Verification
#
# Increasingly, as we design photonic-electronic systems, it is necessary to verify the implemented optical logic. For example, to explore whether the optical switching network implements the desired dynamic optical function, or to compute how much the system thermal performance for a given operation is required.
#
# The point of this script is to demonstrate relevant verification functions. What it should do is to take the optical logic implemented and input some test functions with ideal models and
# compare with the expected optical logic.

# See the example [Analytical MZM Model](./06a_analytical_mzm_model.html) TODO verify link. to first understand the underlying physics behind this numerical implementation, before we discuss a logical verification of the "optical function" applied.

# ## A Simple MZI Case
#
# Let's do a simple review of how an interferometer works and compare it to the simulation we are performing.
#
# Let's consider a simple MZI 2x2 logic with two transmission states. We want to verify that the electronic function switch, effectively switches the optical output between the cross and bar states of the optical transmission function.
#
# For the corresponding model:
#
# Let's assume a switch model unitary. For a given 2x2 input optical switch "X".
#
# In bar a state, in dual rail, transforms an optical input (on an ideal model with $\Delta \phi = \pi$):
# ```
# [[1] ----> [[1]
# [0]]        [0]]
# ```
#
# In cross state, in dual rail, transforms an optical input (on an ideal model with $\Delta \phi = 0$):
# ```
# [[1] ----> [[0]
# [0]]        [1]]
# ```

import functools
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from piel import straight_heater_metal_simple
import piel
import sax

# ## Circuit Construction

# We will explore and compose our switch as we have done in some of the previous examples.

# +
# Parameters directly parametrized from this function call
ideal_resistive_heater = functools.partial(
    straight_heater_metal_simple, ohms_per_square=1
)

ideal_mzi_2x2_2x2_phase_shifter = gf.components.mzi2x2_2x2_phase_shifter(
    straight_x_top=ideal_resistive_heater,
)
ideal_mzi_2x2_2x2_phase_shifter.plot_widget()
# -

# We can extract the optical netlist accordingly.

optical_recursive_netlist = functools.partial(
    gf.get_netlist.get_netlist, exclude_port_types="electrical"
)
switch_netlist = optical_recursive_netlist(ideal_mzi_2x2_2x2_phase_shifter)
# pp.pprint(switch_netlist)

# ## Data Extraction

# These are our optical switch function tests.

valid_input_fock_states = piel.fock_states_only_individual_modes(
    mode_amount=2,
    maximum_photon_amount=1,
    output_type="jax",
)
valid_input_fock_states

# ```
# [Array([[1],
#         [0]], dtype=int32),
#  Array([[0],
#         [1]], dtype=int32)]
# ```

# Let's evaluate our models

verification_models = piel.models.frequency.get_default_models(
    type="optical_logic_verification"
)
verification_models

verification_models["mmi2x2"]()

verification_models["straight_heater_metal_simple"](active_phase_rad=0)

verification_models["straight_heater_metal_simple"](active_phase_rad=jnp.pi)

# Now we need to compose the optical function discretized circuit:

switch_states = [0, jnp.pi]

(
    ideal_mzi_2x2_2x2_phase_shifter_circuit,
    ideal_mzi_2x2_2x2_phase_shifter_circuit_info,
) = sax.circuit(
    netlist=switch_netlist,
    models=verification_models,
)
ideal_mzi_2x2_2x2_phase_shifter_circuit

# Let's evaluate our circuits for both states:

zero_phase_circuit = piel.sax_to_s_parameters_standard_matrix(
    ideal_mzi_2x2_2x2_phase_shifter_circuit(sxt={"active_phase_rad": switch_states[0]}),
    input_ports_order=(
        "o2",
        "o1",
    ),
)
print(zero_phase_circuit)
print(piel.round_complex_array(zero_phase_circuit[0]))

# ```python
# (Array([[0.+0.j, 0.+1.j],
#        [0.+1.j, 0.+0.j]], dtype=complex128), ('o2', 'o1'))
# [[0.+0.j 0.+1.j]
#  [0.+1.j 0.+0.j]]
# ```

pi_phase_circuit = piel.sax_to_s_parameters_standard_matrix(
    ideal_mzi_2x2_2x2_phase_shifter_circuit(sxt={"active_phase_rad": switch_states[1]}),
    input_ports_order=(
        "o2",
        "o1",
    ),
)
print(pi_phase_circuit)
print(piel.round_complex_array(pi_phase_circuit[0]))

# ```python
# (Array([[ 1.000000e+00+6.123234e-17j,  6.123234e-17+0.000000e+00j],
#        [ 6.123234e-17+0.000000e+00j, -1.000000e+00-6.123234e-17j]],      dtype=complex128), ('o2', 'o1'))
# [[ 1.+0.j  0.+0.j]
#  [ 0.+0.j -1.+0.j]]
# ```

# Identify states:
# * You can see it's the cross state because the outputs invert/cross the inputs.
# * You can see it's the bar state because the outputs map the inputs.

# ### Optical Function without $\pi$-phase Applied

raw_output_state_0 = jnp.dot(zero_phase_circuit[0], valid_input_fock_states[0])
output_state_0 = {
    "phase": (switch_states[0],),
    "input_fock_state": piel.convert_array_type(
        valid_input_fock_states[0], piel.tuple_int_type
    ),
    "output_fock_state": piel.absolute_to_threshold(
        raw_output_state_0, output_array_type=piel.tuple_int_type
    ),
}
output_state_0

# ```python
# {'phase': (0,), 'input_fock_state': (1, 0), 'output_fock_state': (0, 1)}
# ```

# You can also compose these type of data in a format that `piel` standardizes in order to implement the functional verification with a nicer helper function:

raw_output_state_1 = jnp.dot(zero_phase_circuit[0], valid_input_fock_states[1])
output_state_1 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(0,),
    input_fock_state_array=valid_input_fock_states[1],
    raw_output_state=raw_output_state_1,
)
output_state_1

# ```python
# {'phase': (0,), 'input_fock_state': (0, 1), 'output_fock_state': (1, 0)}
# ```

# ### Optical Function with $\pi$-phase Applied

raw_output_state_2 = jnp.dot(pi_phase_circuit[0], valid_input_fock_states[0])
output_state_2 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(jnp.pi,),
    input_fock_state_array=valid_input_fock_states[0],
    raw_output_state=raw_output_state_2,
)
output_state_2

# ```python
# {'phase': (3.141592653589793,),
#  'input_fock_state': (1, 0),
#  'output_fock_state': (1, 0)}
# ```

raw_output_state_3 = jnp.dot(pi_phase_circuit[0], valid_input_fock_states[1])
output_state_3 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(jnp.pi,),
    input_fock_state_array=valid_input_fock_states[1],
    raw_output_state=raw_output_state_3,
)
output_state_3

# ```python
# {'phase': (3.141592653589793,),
#  'input_fock_state': (0, 1),
#  'output_fock_state': (0, 1)}
# ```

# ### Formal Verification

# It would be nice to create a little "optical truth table" to implement formal logic verification of our optical circuits, and then compare whether our optical circuit functions are correctly implemented. What we would have is a phase applied, which would be our input, alongside the corresponding fock states, and then we can compute the corresponding fock state output.
#
# It would be really interesting to implement a formal verification protocol like in electronics of this form:

verification_states = [
    {"phase": (0,), "input_fock_state": (1, 0), "output_fock_state": (0, 1)},
    {"phase": (0,), "input_fock_state": (0, 1), "output_fock_state": (1, 0)},
    {
        "phase": (3.141592653589793,),
        "input_fock_state": (1, 0),
        "output_fock_state": (1, 0),
    },
    {
        "phase": (3.141592653589793,),
        "input_fock_state": (0, 1),
        "output_fock_state": (0, 1),
    },
]
target_verification_dataframe = pd.DataFrame(verification_states)
target_verification_dataframe

# |    | phase                | input_fock_state   | output_fock_state   |
# |---:|:---------------------|:-------------------|:--------------------|
# |  0 | (0,)                 | (1, 0)             | (0, 1)              |
# |  1 | (0,)                 | (0, 1)             | (1, 0)              |
# |  2 | (3.141592653589793,) | (1, 0)             | (1, 0)              |
# |  3 | (3.141592653589793,) | (0, 1)             | (0, 1)              |
#

# Now, we can use a convenient `piel` functionality to implement verification of our photonic chip logic with this type of format.
#
# First, we compose our computed truth table:

computed_verification_dataframe = pd.DataFrame(
    [output_state_0, output_state_1, output_state_2, output_state_3]
)
computed_verification_dataframe

# |    | phase                | input_fock_state   | output_fock_state   |
# |---:|:---------------------|:-------------------|:--------------------|
# |  0 | (0,)                 | (1, 0)             | (0, 1)              |
# |  1 | (0,)                 | (0, 1)             | (1, 0)              |
# |  2 | (3.141592653589793,) | (1, 0)             | (1, 0)              |
# |  3 | (3.141592653589793,) | (0, 1)             | (0, 1)              |
#

# Verification just involves checking if the internal values are equivalent. This is straightforward to do using pandas modules.

target_verification_dataframe.compare(computed_verification_dataframe)

target_verification_dataframe.equals(computed_verification_dataframe)

# ```python
# True
# ```
# This can vary for reasons unclear.

# ### Automatic Verification
#
# TODO: One thing I have noticed is that depending on the random configuration of the runner, sometimes the cross and bar states invert on which phase they map. I need to see how to fix that within the computation, if it is even possible.

output_transition_mzi_2x2 = piel.models.logic.electro_optic.get_state_phase_transitions(
    switch_function=ideal_mzi_2x2_2x2_phase_shifter_circuit,
    switch_states=[0, jnp.pi],
    mode_amount=2,
    input_ports_order=("o2", "o1"),
)
output_transition_mzi_2x2

# ```python
# [{'phase': (0,), 'input_fock_state': (1, 0), 'output_fock_state': (0, 1)},
#  {'phase': (0,), 'input_fock_state': (0, 1), 'output_fock_state': (1, 0)},
#  {'phase': (3.141592653589793,),
#   'input_fock_state': (1, 0),
#   'output_fock_state': (1, 0)},
#  {'phase': (3.141592653589793,),
#   'input_fock_state': (0, 1),
#   'output_fock_state': (0, 1)}]
# ```

# We can get the corresponding phase:

# +
print("Current Numerical Implementation")
cross_phase = piel.models.logic.electro_optic.extract_phase(
    output_transition_mzi_2x2, transition_type="cross"
)
print("Cross phase:", cross_phase)

bar_phase = piel.models.logic.electro_optic.extract_phase(
    output_transition_mzi_2x2, transition_type="bar"
)
print("Bar phase:", bar_phase)
# -

# ```
# Current Numerical Implementation
# Cross phase: 0
# Bar phase: 3.141592653589793
# ```

# +
target_output_transition_mzi_2x2 = [
    {"phase": (0,), "input_fock_state": (1, 0), "output_fock_state": (0, 1)},
    {"phase": (0,), "input_fock_state": (0, 1), "output_fock_state": (1, 0)},
    {
        "phase": (3.141592653589793,),
        "input_fock_state": (1, 0),
        "output_fock_state": (1, 0),
    },
    {
        "phase": (3.141592653589793,),
        "input_fock_state": (0, 1),
        "output_fock_state": (0, 1),
    },
]

print("Target Numerical Implementation")
cross_phase = piel.models.logic.electro_optic.extract_phase(
    target_output_transition_mzi_2x2, transition_type="cross"
)
print("Cross phase:", cross_phase)

bar_phase = piel.models.logic.electro_optic.extract_phase(
    target_output_transition_mzi_2x2, transition_type="bar"
)
print("Bar phase:", bar_phase)
# -

# ```
# Target Numerical Implementation
# Cross phase: 0
# Bar phase: 3.141592653589793
# ```

# We can verify the effective transition:

assert output_transition_mzi_2x2 == target_output_transition_mzi_2x2

# ## Switch Fabric Logic Verification

# ### Setup

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2_phase_shifter
import numpy as np
import jax.numpy as jnp
import piel
import sax

piel.visual.activate_piel_styles()

# Now, let's create the s-matrix model circuit:

# +
mzi2x2_2x2_phase_shifter_netlist = mzi2x2_2x2_phase_shifter().get_netlist(
    exclude_port_types="electrical"
)

chain_3_mode_lattice = [
    [mzi2x2_2x2_phase_shifter(), 0],
    [0, mzi2x2_2x2_phase_shifter()],
]

chain_3_mode_lattice_circuit = gf.components.component_lattice_generic(
    network=chain_3_mode_lattice
)
# mixed_switch_circuit.show()
chain_3_mode_lattice_circuit.plot_widget()
# CURRENT TODO: Create a basic chain fabric and verify the logic is implemented properly with binary inputs.

chain_3_mode_lattice = [
    [mzi2x2_2x2_phase_shifter(), 0],
    [0, mzi2x2_2x2_phase_shifter()],
]

chain_3_mode_lattice_circuit = gf.components.component_lattice_generic(
    network=chain_3_mode_lattice
)

# +
chain_3_mode_lattice_circuit_netlist = (
    chain_3_mode_lattice_circuit.get_netlist_recursive(
        exclude_port_types="electrical", allow_multiple=True
    )
)

recursive_composed_required_models = sax.get_required_circuit_models(
    chain_3_mode_lattice_circuit_netlist["component_lattice_gener_3e0da86b"],
    models=piel.models.frequency.get_default_models(),
)


recursive_composed_required_models_0 = sax.get_required_circuit_models(
    chain_3_mode_lattice_circuit_netlist[
        [
            model
            for model in recursive_composed_required_models
            if model.startswith("mzi")
        ][0]
    ],
    models=piel.models.frequency.get_default_models(),
)

recursive_composed_required_models_0

straight_heater_metal_simple = verification_models["straight_heater_metal_simple"]

our_recursive_custom_library = (
    piel.models.frequency.compose_custom_model_library_from_defaults(
        custom_defaults=verification_models,
        custom_models={
            "straight_heater_metal_s_ad3c1693": straight_heater_metal_simple
        },
    )
)
our_recursive_custom_library
# -

(
    chain_3_mode_lattice_circuit_s_parameters,
    chain_3_mode_lattice_circuit_s_parameters_info,
) = sax.circuit(
    netlist=chain_3_mode_lattice_circuit_netlist,
    models=our_recursive_custom_library,
)

# Let's explore the four states of our switch lattice in an explicit form:

# +
# TODO work out why it needs to be inverted, it's an implementation problem, continuing for now.
bar_bar_state = piel.sax_to_s_parameters_standard_matrix(
    chain_3_mode_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": bar_phase}},
        mzi_2={"sxt": {"active_phase_rad": bar_phase}},
    )
)[0]
bar_bar_state = jnp.abs(piel.round_complex_array(bar_bar_state))

cross_bar_state = piel.sax_to_s_parameters_standard_matrix(
    chain_3_mode_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": cross_phase}},
        mzi_2={"sxt": {"active_phase_rad": bar_phase}},
    ),
    round_int=True,
    to_absolute=True,
)[0]


bar_cross_state = piel.sax_to_s_parameters_standard_matrix(
    chain_3_mode_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": bar_phase}},
        mzi_2={"sxt": {"active_phase_rad": cross_phase}},
    ),
    round_int=True,
    to_absolute=True,
)[0]

cross_cross_state = piel.sax_to_s_parameters_standard_matrix(
    chain_3_mode_lattice_circuit_s_parameters(
        mzi_1={"sxt": {"active_phase_rad": cross_phase}},
        mzi_2={"sxt": {"active_phase_rad": cross_phase}},
    ),
    round_int=True,
    to_absolute=True,
)[0]


print("bar_bar_state")
print(bar_bar_state)
print("cross_bar_state")
print(cross_bar_state)
print("bar_cross_state")
print(bar_cross_state)
print("cross_cross_state")
print(cross_cross_state)
# -

# ```
# bar_bar_state
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
# cross_bar_state
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# bar_cross_state
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]
# cross_cross_state
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# ```

# ### Validation

# Let's consider the case that we have an optical input at the top of the switching network, and we simply want to route it to the bottom output mode. We will the 2x2 MZI switch state described in the previous section to understand the encoded logic.
#
# TODO make an illustration here.

top_optical_input = jnp.array([[1], [0], [0]])
top_optical_input

# ```python
# Array([[1],
#        [0],
#        [0]], dtype=int32)
# ```

jnp.abs(jnp.dot(bar_bar_state, top_optical_input))

# ```python
# Array([[1.],
#        [0.],
#        [0.]], dtype=float32)
# ```

jnp.abs(jnp.dot(cross_bar_state, top_optical_input))

# ```python
# Array([[0.],
#        [1.],
#        [0.]], dtype=float32)
# ```

jnp.abs(jnp.dot(bar_cross_state, top_optical_input))

# ```python
# Array([[1.],
#        [0.],
#        [0.]], dtype=float32)
# ```

jnp.abs(jnp.dot(cross_cross_state, top_optical_input))

# ```python
# Array([[0.],
#        [0.],
#        [1.]], dtype=float32)
# ```

# ## Further Analytical Modelling
#
# Let's consider how a switching network behaves symbolically. Say we have two switches in a chain, illustrated by this format:

chain_mode_3 = np.array(
    [
        [
            "X",
            0,
        ],
        [0, "X"],
    ]
)
chain_mode_3_switch_position_list = (
    piel.models.logic.photonic.compose_switch_position_list(network=chain_mode_3)
)
chain_mode_3, chain_mode_3_switch_position_list

# Let's consider the "X" state can only have two possible states, cross and bar which are represented by the angle applied, (0 -> 0, bar) and (1 -> $\pi$, cross).
#
# If we have a fock state `[[1], [0], [0]]` inputted onto the switch lattice, we want it to route out the photon accordingly at the bottom mode index 2, third waveguide. Accordingly, the top-most switch needs to cross and the bottom most needs to bar in order to achieve this function.
#
#
# We can try a little analytical simulator accordingly. Each "switch" state gets replaced by a 2x2 transmission matrix for each specific state, and concatenated to build the corresponding state of the system.

piel.models.logic.electro_optic.get_state_phase_transitions(
    switch_function=chain_3_mode_lattice_circuit_s_parameters,
    switch_states=[0, np.pi],
    mode_amount=3,
)

# ```python
# [{'phase': (0,),
#   'input_fock_state': (1, 0, 0),
#   'output_fock_state': (0, 0, 1)},
#  {'phase': (0,),
#   'input_fock_state': (0, 1, 0),
#   'output_fock_state': (1, 0, 0)},
#  {'phase': (0,),
#   'input_fock_state': (0, 0, 1),
#   'output_fock_state': (0, 1, 0)},
#  {'phase': (3.141592653589793,),
#   'input_fock_state': (1, 0, 0),
#   'output_fock_state': (0, 0, 1)},
#  {'phase': (3.141592653589793,),
#   'input_fock_state': (0, 1, 0),
#   'output_fock_state': (1, 0, 0)},
#  {'phase': (3.141592653589793,),
#   'input_fock_state': (0, 0, 1),
#   'output_fock_state': (0, 1, 0)}]
# ```
