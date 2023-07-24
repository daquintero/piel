# # Quantum Integration Basics

# $$\newcommand{\ket}[1]{\left|{#1}\right\rangle}$$
# $$\newcommand{\bra}[1]{\left\langle{#1}\right|}$$
#
# One interesting thing to explore would be quantum state evolution through a unitary matrix composed from a physical photonic network that we could model. We will explore in this example how to integrate `sax` and `qutip`.

import gdsfactory as gf
import sax
import piel
import qutip as qp

# ## Photonics Circuit to Unitary

# We follow the same process as the previous examples:

switch_circuit = gf.components.component_lattice_generic()
switch_circuit.show()
switch_circuit.plot_widget()

# ![default_switch_circuit_plot_widget](../_static/img/examples/03_sax_basics/default_switch_circuit_plot_widget.PNG)

recursive_netlist = switch_circuit.get_netlist_recursive()
switch_circuit_model, switch_circuit_model_info = sax.circuit(
    netlist=recursive_netlist,
    models=piel.models.frequency.get_default_models(),
)
default_state_s_parameters = switch_circuit_model()

# We convert from the `sax` unitary to an ideal "unitary" that can be inputted into a `qutip` model. Fortunately, `piel` has got you covered:
# It is important to note some inherent assumptions and limitations of the translation process.

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

import numpy as np

np.asarray(s_parameters_standard_matrix)

# We can explore some properties of this matrix:

s_parameters_standard_matrix.shape

# ## Quantum Unitary Representation

# + active=""
# # TODO verify validity, I am a bit unconvinced re loss and identities.
#
# See docs/microservices/dependencies/qutip for further theory and validation. TODO link.
# -

qutip_qobj = piel.standard_s_parameters_to_qutip_qobj(s_parameters_standard_matrix)

qutip_qobj

# ![example_qutip_unitary](../_static/img/examples/05_quantum_integration_basics/example_qutip_unitary.PNG)

qutip_qobj.check_isunitary()

qutip_qobj.dims

qutip_qobj.eigenstates

# ## Sub Circuit Unitary Analysis

# We can also integrated the `piel` toolchain, with another set of packages for quantum photonic system design such as those provided by `XanaduAI`. We will use their `thewalrus` package to calculate the permanent of our matrix. For example, we can do this for our full circuit unitary:

piel.sax_circuit_permanent(default_state_s_parameters)

# We might want to calculate the permanent of subsections of the larger unitary to calculate certain operations probability:

s_parameters_standard_matrix.shape

# For, example, we need to just calculate it for the first submatrix component, or a particular switch unitary within a larger circuit. This would be indexed when starting from the first row and column as `start_index` = (0,0) and `stop_index` = (`unitary_size`, `unitary_size`). Note that an error will be raised if a non-unitary matrix is inputted. Some examples are:

our_subunitary = piel.subunitary_selection_on_range(
    s_parameters_standard_matrix, stop_index=(1, 1), start_index=(0, 0)
)
our_subunitary

# ```python
# Array([[ 0.40105772+0.49846345j, -0.45904815-0.197149j  ],
#        [-0.4590482 -0.197149j  , -0.8361797 +0.13278401j]],      dtype=complex64)
# ```

# We can now calculate the permanent of this submatrix:

piel.unitary_permanent(our_subunitary)

# ```python
# ((-0.2296868-0.18254918j), 0.0)
# ```

# ## Fock State Evolution Probability

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
    unitary_matrix=s_parameters_standard_matrix,
    rows_index=final_fock_state_indices,
    columns_index=initial_fock_state_indices,
)

# ```python~
# Array([[ 0.40105772+0.49846345j,  0.4000432 +0.38792986j],
#        [ 0.40004322+0.3879298j , -0.5810837 -0.31133226j]],      dtype=complex64)
# ```

# We can now extract the transition amplitude probability accordingly:

piel.fock_transition_probability_amplitude(
    initial_fock_state=initial_fock_state,
    final_fock_state=final_fock_state,
    unitary_matrix=s_parameters_standard_matrix,
)

# ```python
# Array(-0.06831534-0.10413378j, dtype=complex64)
# ```
#
# TODO this is not numerically right but the functions are fine because we need to verify the unitary-ness of the model composed matrices.
