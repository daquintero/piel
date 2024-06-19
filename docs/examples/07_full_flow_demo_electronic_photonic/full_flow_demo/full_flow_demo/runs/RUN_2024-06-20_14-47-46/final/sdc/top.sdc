###############################################################################
# Created by write_sdc
###############################################################################
current_design top
###############################################################################
# Timing Constraints
###############################################################################
create_clock -name None -period 15.0000 
set_clock_uncertainty 0.2500 None
set_input_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {input_fock_state_str[0]}]
set_input_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {input_fock_state_str[1]}]
set_input_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {input_fock_state_str[2]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_0[0]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_0[1]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_0[2]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_0[3]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_0[4]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_1[0]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_1[1]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_1[2]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_1[3]}]
set_output_delay 3.0000 -clock [get_clocks {None}] -add_delay [get_ports {bit_phase_1[4]}]
###############################################################################
# Environment
###############################################################################
set_load -pin_load 0.0334 [get_ports {bit_phase_0[4]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_0[3]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_0[2]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_0[1]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_0[0]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_1[4]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_1[3]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_1[2]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_1[1]}]
set_load -pin_load 0.0334 [get_ports {bit_phase_1[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {input_fock_state_str[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {input_fock_state_str[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {input_fock_state_str[0]}]
###############################################################################
# Design Rules
###############################################################################
set_max_transition 0.7500 [current_design]
set_max_capacitance 0.2000 [current_design]
set_max_fanout 6.0000 [current_design]
