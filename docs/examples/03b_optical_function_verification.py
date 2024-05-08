# # Optical Function Verification
#
# Increasingly, as we design photonic-electronic systems, it is necessary to verify the implemented optical logic. For example, to explore whether the optical switching network implements the desired dynamic optical function, or to compute how much the system thermal performance for a given operation is required.
#
# The point of this script is to demonstrate relevant verification functions. What it should do is to take the optical logic implemented and input some test functions with ideal models and
# compare with the expected optical logic.
#
# Let's consider a simple MZI 2x2 logic with two transmission states. We want to verify that the electronic function switch, effectively switches the optical output between the cross and bar states of the optical transmission function.
#
# For the corresponding model:
#
# Let's assume a switch model unitary. For a given 2x2 input optical switch "X". In bar state, in dual rail, transforms an optical input:
# ```
# [[1] ----> [[1]
# [0]]        [0]]
# ```
#
# In cross state, in dual rail, transforms an optical input:
# ```
# [[1] ----> [[0]
# [0]]        [1]]
# ```

# +
import functools
from itertools import product
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from piel import straight_heater_metal_simple
import piel
import pprint as pp
import sax
import random

jax.random.key(0)
np.random.seed(0)
# -

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

verification_models = piel.models.frequency.get_default_models(type="optical_logic_verification")
verification_models

verification_models["mmi2x2"]()

verification_models["straight_heater_metal_simple"](active_phase_rad=0)

verification_models["straight_heater_metal_simple"](active_phase_rad=1)

verification_models["straight_heater_metal_simple"](active_phase_rad=jnp.pi)

# Now we need to compose the optical function discretized circuit:

switch_states = [0, jnp.pi]

ideal_mzi_2x2_2x2_phase_shifter_circuit, ideal_mzi_2x2_2x2_phase_shifter_circuit_info = sax.circuit(
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
zero_phase_circuit

pi_phase_circuit = piel.sax_to_s_parameters_standard_matrix(
    ideal_mzi_2x2_2x2_phase_shifter_circuit(sxt={"active_phase_rad": switch_states[1]}),
    input_ports_order=(
        "o2",
        "o1",
    ),
)
pi_phase_circuit

# Identify states: 
# * You can see it's the cross state because the outputs invert/cross the inputs.
# * You can see it's the bar state because the outputs map the inputs.

# ### Optical Function without $\pi$-phase Applied

raw_output_state_0 = jnp.dot(zero_phase_circuit[0], valid_input_fock_states[0])
output_state_0 = {
    "phase": (switch_states[0],),
    "input_fock_state": piel.convert_array_type(valid_input_fock_states[0], piel.tuple_int_type),
    "output_fock_state": piel.absolute_to_threshold(raw_output_state_0, output_array_type=piel.tuple_int_type),
}
output_state_0

# ```python
# {'phase': (0,), 
#  'input_fock_state': (1, 0),
#  'output_fock_state': (1, 0)}
# ```

# You can also compose these type of data in a format that `piel` standardizes in order to implement the functional verification with a nicer helper function:

raw_output_state_1 = jnp.dot(zero_phase_circuit[0], valid_input_fock_states[1])
output_state_1 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(0,),
    input_fock_state_array=valid_input_fock_states[1],
    raw_output_state=raw_output_state_1
)
output_state_1

# ```python
# {'phase': (0,), 
#  'input_fock_state': (0, 1),
#  'output_fock_state': (0, 1)}
# ```

# ### Optical Function with $\pi$-phase Applied

raw_output_state_2 = jnp.dot(pi_phase_circuit[0], valid_input_fock_states[0])
output_state_2 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(jnp.pi,),
    input_fock_state_array=valid_input_fock_states[0],
    raw_output_state=raw_output_state_2
)
output_state_2

# ```python
# {'phase': (3.141592653589793,),
#  'input_fock_state': (1, 0),
#  'output_fock_state': (0, 1)}
# ```

raw_output_state_3 = jnp.dot(pi_phase_circuit[0], valid_input_fock_states[1])
output_state_3 = piel.models.logic.electro_optic.format_electro_optic_fock_transition(
    switch_state_array=(jnp.pi,),
    input_fock_state_array=valid_input_fock_states[1],
    raw_output_state=raw_output_state_3
)
output_state_3

# ```python
# {'phase': (3.141592653589793,),
#  'input_fock_state': (0, 1),
#  'output_fock_state': (1, 0)}
# ```

# ### Formal Verification

# It would be nice to create a little "optical truth table" to implement formal logic verification of our optical circuits, and then compare whether our optical circuit functions are correctly implemented. What we would have is a phase applied, which would be our input, alongside the corresponding fock states, and then we can compute the corresponding fock state output.
#
# It would be really interesting to implement a formal verification protocol like in electronics of this form:

verification_states = [
    {
        "phase": (0,),
        "input_fock_state": (1, 0),
        "output_fock_state": (1, 0),
    },
    {
        "phase": (0,),
        "input_fock_state": (0, 1),
        "output_fock_state": (0, 1),
    },
    {
        "phase": (jnp.pi,),
        "input_fock_state": (1, 0),
        "output_fock_state": (0, 1),
    },
    {
        "phase": (jnp.pi,),
        "input_fock_state": (0, 1),
        "output_fock_state": (1, 0),
    },  
]
target_verification_dataframe = pd.DataFrame(verification_states)
target_verification_dataframe

# |    | phase                | input_fock_state | output_fock_state |
# |---:|:---------------------|:-----------------|:------------------|
# |  0 | (0,)                 | (1, 0)           | (1, 0)            |
# |  1 | (0,)                 | (0, 1)           | (0, 1)            |
# |  2 | (3.141592653589793,) | (1, 0)           | (0, 1)            |
# |  3 | (3.141592653589793,) | (0, 1)           | (1, 0)            |
#

# Now, we can use a convienient `piel` functionality to implement verification of our photonic chip logic with this type of format.
#
# First, we compose our computed truth table:

computed_verification_dataframe = pd.DataFrame([output_state_0, output_state_1, output_state_2, output_state_3])
computed_verification_dataframe

# |    | phase                | input_fock_state | output_fock_state |
# |---:|:---------------------|:-----------------|:------------------|
# |  0 | (0,)                 | (1, 0)           | (1, 0)            |
# |  1 | (0,)                 | (0, 1)           | (0, 1)            |
# |  2 | (3.141592653589793,) | (1, 0)           | (0, 1)            |
# |  3 | (3.141592653589793,) | (0, 1)           | (1, 0)            |
#

# Verification just involves checking if the internal values are equivalent. This is straightforward to do using pandas modules.

target_verification_dataframe.compare(computed_verification_dataframe)

target_verification_dataframe.equals(computed_verification_dataframe)

# ```python
# True
# ```

# WARNING: One thing I have noticed is that depending on the random configuration of the runner, sometimes the cross and bar states invert on which phase they map. I need to see how to fix that within the computation, if it is even possible.

# ## Further Analytical Modelling
#
# Let's consider how a switching network behaves symbolically. Say we have two switches in a chain, illustrated by this format:

chain_mode_3 = np.array([['X', 0,],
                         [0, 'X']])
chain_mode_3_switch_position_list = piel.models.logic.photonic.compose_switch_position_list(
    network=chain_mode_3
)
chain_mode_3, chain_mode_3_switch_position_list


# Let's consider the "X" state can only have two possible states, cross and bar which are represented by the angle applied, (0 -> 0, bar) and (1 -> $\pi$, cross). 
#
# If we have a fock state `[[1], [0], [0]]` inputted onto the switch lattice, we want it to route out the photon accordingly at the bottom mode index 2, third waveguide. Accordingly, the top-most switch needs to cross and the bottom most needs to bar in order to achieve this function.
#
#
# We can try a little analytical simulator accordingly. Each "switch" state gets replaced by a 2x2 transmission matrix for each specific state, and concatenated to build the correponding state of the system. 

def a(
    switch_network: list[list]
    states: tuple = (0,1)
):




