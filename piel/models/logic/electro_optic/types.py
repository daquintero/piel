"""
This is the standard format of a corresponding output state for a given input state in the electro-optic model:

output_state_0 = {
    "phase": (switch_states[0],),
    "input_fock_state": piel.convert_array_type(valid_input_fock_states[0], "tuple"),
    "output_fock_state": piel.absolute_to_threshold(raw_output_state_0, output_array_type="tuple"),
}
"""
electro_optic_fock_state_type = {
    "phase": tuple[float, ...],
    "input_fock_state": tuple[int, ...],
    "output_fock_state": tuple[int, ...],
}
