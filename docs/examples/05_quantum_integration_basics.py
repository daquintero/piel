# # Quantum Integration Basics

# One interesting thing to explore would be quantum state evolution through a unitary matrix composed from a physical photonic network that we could model. We will explore in this example how to integrate `sax` and `qutip`.

import gdsfactory as gf  # NOQA : F401
import sax  # NOQA : F401
import piel  # NOQA : F401
import qutip  # NOQA : F401

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

# ```
# array([[ 0.40117208+0.49838198j, -0.45906927-0.19706765j,
#          0.00184268+0.17482864j,  0.40013665+0.38783803j],
#        [-0.45906927-0.19706765j, -0.83617032+0.13289595j,
#         -0.03938958-0.0381789j , -0.17480098+0.0036143j ],
#        [ 0.00184268+0.17482864j, -0.03938958-0.0381789j ,
#         -0.85361747+0.11598505j,  0.11497926-0.45944187j],
#        [ 0.40013665+0.38783803j, -0.17480098+0.0036143j ,
#          0.11497926-0.45944187j, -0.58117378-0.31118139j]])
# ```

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

# ## Calculating the Permanent

# We can also integrated the `piel` toolchain, with another set of packages for quantum photonic system design such as those provided by `XanaduAI`. We will use their `thewalrus` package to calculate the permanent of our matrix.

piel.sax_circuit_permanent(default_state_s_parameters)

# The way this works is straightforward:

# We might want to calculate the permanent of subsections of the larger unitary to calculate certain operations probability:

s_parameters_standard_matrix.shape

# For, example, we might want to just calculate it for the first two input modes. This would be indexed when starting from the first row and column as `start_index` = (0,0) and `stop_index` = (`unitary_size`, `unitary_size`). Note that an error will be raised if a non-unitary matrix is inputted. Some examples are:

s_parameters_standard_matrix

import jax.numpy as jnp

jax_array = jnp.array(s_parameters_standard_matrix)
jax_array

jax_array.at[jnp.array([0, 1])].get()

jnp.ndarray
