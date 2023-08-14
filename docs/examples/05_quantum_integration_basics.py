# # Quantum Integration Basics

# $$\newcommand{\ket}[1]{\left|{#1}\right\rangle}$$
# $$\newcommand{\bra}[1]{\left\langle{#1}\right|}$$
#
# One interesting thing to explore would be quantum state evolution through a unitary matrix composed from a physical photonic network that we could model. We will explore in this example how to integrate `sax` and `qutip`.

import gdsfactory as gf
import sax
import piel
import qutip as qp

# ## Quantum Unitary Representation

# One of the main complexities of modelling a quantum photonic system is that loss is a killer. It is a killer far more than we understand in classical systems. When we model unitary operations, they are lossless. When loss occurs on quantum photonic operations, it translates into Pauli gate errors, or less efficient resource state generation models. This means that we need to model our phase separate from our target logical operations that implement our quantum circuit or specific components accordingly. So, how we model our circuits, essentially depends on the methodology we compose our circuit models. To model a quantum network, we need to compose our circuit connectivity using unitary models. However, to model loss and optical phase variations, we need to then model our circuit with more realstic physical models. This will allow us to extract different aspects of circuit information accordingly.
#
# See docs/microservices/dependencies/qutip for further theory and validation. TODO link.

# ### Starting off from Composed Circuit

switch_circuit = gf.components.component_lattice_generic()
switch_circuit.show()
switch_circuit.plot_widget()

# ![default_switch_circuit_plot_widget](../_static/img/examples/03_sax_basics/default_switch_circuit_plot_widget.PNG)

switch_circuit.get_netlist_recursive().keys()

# ### Quantum Models

# Let's first check that the quantum models in which we will compose our circuit are actually unitary, otherwise the composed circuit will not be unitary. Note that a circuit being unitary means: $U^\dagger U = 1$ where $U^\dagger$ is the conjugate transpose of the unitary $U$. This is inherently checked in `qutip`. Basically, what it means is that a unitary operation is reversible in time, and that energy is not lost.

quantum_models = piel.models.frequency.get_default_models(type="quantum")
quantum_models["mmi2x2"]()

mmi2x2_qobj = piel.sax_to_ideal_qutip_unitary(
    quantum_models["mmi2x2"](), input_ports_order=("o1", "o2")
)
mmi2x2_qobj.check_isunitary()

# We follow the same process as the previous examples, but we use lossless models for the circuit composition.

recursive_netlist = switch_circuit.get_netlist_recursive()
switch_circuit_model_quantum, switch_circuit_model_quantum_info = sax.circuit(
    netlist=recursive_netlist,
    models=piel.models.frequency.get_default_models(type="quantum"),
)
default_state_unitary = switch_circuit_model_quantum()

# We convert from the `sax` unitary to an ideal "unitary" that can be inputted into a `qutip` model. Fortunately, `piel` has got you covered:
# It is important to note some inherent assumptions and limitations of the translation process.

(
    unitary_matrix,
    input_ports_index_tuple_order,
) = piel.sax_to_s_parameters_standard_matrix(default_state_unitary)
unitary_matrix

# ```python
# Array([[ 0.       +0.j       ,  0.       +0.j       ,
#          0.       +0.j       ,  0.       -0.9999998j],
#        [ 0.       +0.j       , -0.9999999+0.j       ,
#          0.       +0.j       ,  0.       +0.j       ],
#        [ 0.       +0.j       ,  0.       +0.j       ,
#         -0.9999999+0.j       ,  0.       +0.j       ],
#        [ 0.       -0.9999998j,  0.       +0.j       ,
#          0.       +0.j       ,  0.       +0.j       ]], dtype=complex64)
# ```

# ### Translating to Qutip

import qutip

switch_circuit_qobj = piel.standard_s_parameters_to_qutip_qobj(unitary_matrix)
switch_circuit_qobj

# ![example_qutip_unitary](../_static/img/examples/05_quantum_integration_basics/example_qutip_unitary.PNG)

switch_circuit_qobj.check_isunitary()

switch_circuit_qobj.dims

# ### Fock State Evolution Probability

# Say, we want to calculate the evolution of a Fock state input through our photonic circuit. The initial Fock state is defined as $\ket{f_1} = \ket{j_1, j_2, ... j_N}$ and transitions to $\ket{f_2} = \ket{j_1^{'}, j_2^{'}, ... j_N^{'}}$. The evolution of this state through our circuit with unitary $U$ is defined by the subunitary $U_{f_1}^{f_2}$.

# Let us define an example four-mode multi-photon Fock state using `qutip` in the state $\ket{f_1} = \ket{1001}$

initial_fock_state = qp.fock(4, 0) + qp.fock(4, 3)
initial_fock_state

final_fock_state = qp.fock(4, 1) + qp.fock(4, 2)
final_fock_state

# The subunitary $U_{f_1}^{f_2}$ is composed from the larger unitary by selecting the rows from the output state Fock state occupation of $\ket{f_2}$, and columns from the input $\ket{f_1}$. In our case, we need to select the columns indexes $(0,3)$ and rows indexes $(1,2)$.
#
# If we only consider an photon number of 1 in the particular Fock state, then we can describe the transition probability amplitude to be equivalent to the permanent:
#
# $$
# a(\ket{f_1} \to \ket{f_2}) = \text{per}(U_{f_1}^{f_2})
# $$
#
# If we consider a photon number of more than one for the transition Fock states, then the Permanent needs to be normalised. The probability amplitude for the transition is described as:
# $$
# a(\ket{f_1} \to \ket{f_2}) = \frac{\text{per}(U_{f_1}^{f_2})}{\sqrt{(j_1! j_2! ... j_N!)(j_1^{'}! j_2^{'}! ... j_N^{'}!)}}
# $$
#
# TODO review Jeremy's thesis citations


# However, we want to explore the probability amplitude of a Fock state transition. Given that our Fock state can have any photon number index, we need to select subsections of the unitary matrix that affect the photon path as described in the algorithm above. Let's implement this functionality based on our `qutip`-defined Fock states.

piel.fock_state_to_photon_number_factorial(initial_fock_state)

# We can analyse this for a multi-photon Fock state:

example_multiphoton_fock_state = (
    qp.fock(4, 1) + qp.fock(4, 2) + qp.fock(4, 2) + qp.fock(4, 2) + qp.fock(4, 2)
)
example_multiphoton_fock_state

piel.fock_state_to_photon_number_factorial(example_multiphoton_fock_state)

# In order to implement the algorithm above, we need to determine the indexes we need to extract for the particular Fock state that we are implementing too.

initial_fock_state_indices = piel.fock_state_nonzero_indexes(initial_fock_state)
initial_fock_state_indices

# ```python
# (0, 3)
# ```

final_fock_state_indices = piel.fock_state_nonzero_indexes(final_fock_state)
final_fock_state_indices

# ```python
# (1, 2)
# ```

# We can extract the section of the unitary that corresponds to this Fock state transition. Note that based on (TODO cite Jeremy), the initial Fock state corresponds to the columns of the unitary and the final Fock states corresponds to the rows of the unitary.

piel.subunitary_selection_on_index(
    unitary_matrix=unitary_matrix,
    rows_index=final_fock_state_indices,
    columns_index=initial_fock_state_indices,
)

# ```python~
# Array([[0.+0.j       , 0.-0.9999998j],
#        [0.-0.9999998j, 0.+0.j       ]], dtype=complex64)
# ```

# We can now extract the transition amplitude probability accordingly:

piel.fock_transition_probability_amplitude(
    initial_fock_state=initial_fock_state,
    final_fock_state=final_fock_state,
    unitary_matrix=unitary_matrix,
)

# ```python
# Array(-0.99999964+0.j, dtype=complex64)
# ```

# ### Fock-State Generation

# It might be desired to generate a large amount of Fock-states to evaluate how the system behaves when performing a particular operation. `piel` provides a few handy functions. For an determined amount of modes and maximum photon number on each state, we can generate all the possible Fock states in `qutip` notation.

input_fock_states = piel.all_fock_states_from_photon_number(
    mode_amount=4, photon_amount=1
)
input_fock_states[10]

# Quantum object: dims = [[4], [1]], shape = (4, 1), type = ket $\left(\begin{matrix}1.0\\0.0\\1.0\\0.0\\\end{matrix}\right)$

piel.all_fock_states_from_photon_number(
    mode_amount=4, photon_amount=1, output_type="jax"
)

# ### Sub Circuit Unitary Analysis

# We can also integrated the `piel` toolchain, with another set of packages for quantum photonic system design such as those provided by `XanaduAI`. We will use their `thewalrus` package to calculate the permanent of our matrix. For example, we can do this for our full circuit unitary:

piel.sax_circuit_permanent(default_state_unitary)

# We might want to calculate the permanent of subsections of the larger unitary to calculate certain operations probability:

unitary_matrix.shape

# For, example, we need to just calculate it for the first submatrix component, or a particular switch unitary within a larger circuit. This would be indexed when starting from the first row and column as `start_index` = (0,0) and `stop_index` = (`unitary_size`, `unitary_size`). Note that an error will be raised if a non-unitary matrix is inputted. Some examples are:

our_subunitary = piel.subunitary_selection_on_range(
    unitary_matrix, stop_index=(1, 1), start_index=(0, 0)
)
our_subunitary

# ```python
# Array([[ 0.       +0.j,  0.       +0.j],
#        [ 0.       +0.j, -0.9999999+0.j]], dtype=complex64)
# ```

# We can now calculate the permanent of this submatrix:

piel.unitary_permanent(our_subunitary)

# ```python
# (0j, 0.0)
# ```

# ## Lossy Models

# What we will do now is explore how our circuit behaves when composing it with more realistic physical models.

recursive_netlist = switch_circuit.get_netlist_recursive()
switch_circuit_model_classical, switch_circuit_model_classical_info = sax.circuit(
    netlist=recursive_netlist,
    models=piel.models.frequency.get_default_models(type="classical"),
)
default_state_s_parameters = switch_circuit_model_classical()

# Let's first convert to a standard S-Parameter matrix:

(
    s_parameters_standard_matrix,
    input_ports_index_tuple_order,
) = piel.sax_to_s_parameters_standard_matrix(default_state_s_parameters)
s_parameters_standard_matrix

# ```python
# Array([[ 0.40105772+0.49846345j, -0.45904815-0.197149j  ,
#          0.00180554+0.17483076j,  0.4000432 +0.38792986j],
#        [-0.4590482 -0.197149j  , -0.8361797 +0.13278401j,
#         -0.03938162-0.03818914j, -0.17480364+0.00356933j],
#        [ 0.00180554+0.17483076j, -0.03938162-0.03818914j,
#         -0.8536251 +0.11586684j,  0.11507235-0.45943272j],
#        [ 0.40004322+0.3879298j , -0.17480363+0.00356933j,
#          0.11507231-0.45943272j, -0.5810837 -0.31133226j]],      dtype=complex64)
# ```

# We can explore some properties of this matrix:

s_parameters_standard_matrix.shape
