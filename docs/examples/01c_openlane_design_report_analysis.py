# # OpenLane Run Output Analysis

import piel

# First, we get the directory of the latest run:

latest_run_output = piel.find_design_run(
    design_directory="inverter",
)
latest_run_output

# We get all the timing STA design files accordingly.

run_output_sta_file_list = piel.get_all_timing_sta_files(
    run_directory=latest_run_output
)
run_output_sta_file_list

# Say we want to explore the output of one particular timing file. We can extract all the timing data accordingly:

file_lines_data = piel.get_frame_lines_data(file_path=run_output_sta_file_list[0])
timing_data = piel.get_all_timing_data_from_file(file_path=run_output_sta_file_list[0])[
    1
]
timing_data

# We can extract the propagation delay from the input and output frame accordingly.

piel.calculate_propagation_delay_from_file(file_path=run_output_sta_file_list[0])[0]
