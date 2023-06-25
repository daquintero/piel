# # SAX Co-simulation Basic Example

# In this example, we will explore different methodologies of mapping electronic signals to photonic operations. We will start by an ideal basic example, and explore the complexity of how these systems can be interconnected accordingly. We will explore different encodings of transformations between electronic simulation implementations and corresponding photonic solutions.

# In order to solve a photonic circuit using `sax`, we first need a physical netlist of our circuit that represents the inputs and outputs that we care about of our circuit.

# We begin by importing a parametric circuit from `gdsfactory`:
import gdsfactory as gf
from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2

example_component_lattice = [
    [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=80.0)],
    [0, mzi2x2_2x2(delta_length=50.0), 0],
    [mzi2x2_2x2(delta_length=100.0), 0, mzi2x2_2x2_phase_shifter()],
]

switch_circuit = gf.components.generic_component_lattice(
    physical_network=example_component_lattice
)
switch_circuit

# Now we need to include our device models, we will start with basic ones and expand from that.
