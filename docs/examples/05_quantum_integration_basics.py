# # Quantum Integration Basics

# One interesting thing to explore would be quantum state evolution through a unitary matrix composed from a physical photonic network that we could model. We will explore in this example how to integrate `sax` and `qutip`.

import gdsfactory as gf  # NOQA : F401
import sax  # NOQA : F401
import piel  # NOQA : F401
import qutip  # NOQA : F401

# ## Photonics Circuit to Unitary

# We follow the same process as the previous examples:

switch_circuit = gf.components.component_lattice_generic()
recursive_netlist = switch_circuit.get_netlist_recursive()
switch_circuit_model, switch_circuit_model_info = sax.circuit(
    netlist=recursive_netlist,
    models=piel.models.frequency.photonic.get_default_models(),
)
default_state_s_parameters = switch_circuit_model()

# We convert from the `sax` unitary to an ideal "unitary" that can be inputted into a `qutip` model. Fortunately, `piel` has got you covered:
# It is important to note some inherent assumptions and limitations of the translation process.


qutip_unitary = piel.sax_s_dict_to_ideal_qutip_unitary(default_state_s_parameters)
