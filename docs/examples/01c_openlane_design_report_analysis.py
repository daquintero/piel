# # OpenLane Run Output Analysis

import piel

# First, we get the directory of the latest run:

latest_run_output = piel.find_design_run(
    design_directory="./designs" / piel.return_path("inverter"),
)
latest_run_output

# + active=""
# WindowsPath('designs/inverter/runs/RUN_2023.06.22_15.40.17')
# -

# We get all the timing STA design files accordingly.

run_output_sta_file_list = piel.get_all_timing_sta_files(
    run_directory=latest_run_output
)
run_output_sta_file_list

# ```
# ['C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\placement\\10-dpl_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\placement\\10-dpl_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\13-rsz_design_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\13-rsz_design_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\15-rsz_timing_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\15-rsz_timing_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\18-grt_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\routing\\18-grt_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\28-rcx_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\28-rcx_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\23-mca\\rcx_min_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\23-mca\\rcx_min_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\25-mca\\rcx_max_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\25-mca\\rcx_max_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\27-mca\\rcx_nom_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\signoff\\27-mca\\rcx_nom_sta.min.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\synthesis\\2-syn_sta.max.rpt',
#  'C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\inverter\\runs\\RUN_2023.06.22_15.40.17\\reports\\synthesis\\2-syn_sta.min.rpt']
# ```

# Say we want to explore the output of one particular timing file. We can extract all the timing data accordingly:

file_lines_data = piel.get_frame_lines_data(file_path=run_output_sta_file_list[0])
timing_data = piel.get_all_timing_data_from_file(file_path=run_output_sta_file_list[0])[
    1
]
timing_data

# |    | Fanout   | Cap      | Slew     | Delay    | Time     | Direction   | Description                           | net_type                  | net_name              |
# |---:|:---------|:---------|:---------|:---------|:---------|:------------|:--------------------------------------|:--------------------------|:----------------------|
# |  0 | nan      | nan      | 0.00     | 0.00     | 0.00     | nan         | clock __VIRTUAL_CLK__ (rise edge)     | rise edge                 | clock __VIRTUAL_CLK__ |
# |  1 | nan      | nan      | nan      | 0.00     | 0.00     | nan         | clock network delay (ideal)           | ideal                     | clock network delay   |
# |  2 | nan      | nan      | nan      | 2.00     | 2.00     | ^           | input external delay                  | nan                       | nan                   |
# |  3 | nan      | nan      | 0.02     | 0.01     | 2.01     | ^           | in (in)                               | in                        | in                    |
# |  4 | 1        | 0.00     | nan      | nan      | nan      | nan         | in (net)                              | net                       | in                    |
# |  5 | nan      | nan      | 0.02     | 0.00     | 2.01     | ^           | input1/A (sky130_fd_sc_hd__buf_1)     | sky130_fd_sc_hd__buf_1    | input1/A              |
# |  6 | nan      | nan      | 0.11     | 0.13     | 2.14     | ^           | input1/X (sky130_fd_sc_hd__buf_1)     | sky130_fd_sc_hd__buf_1    | input1/X              |
# |  7 | 1        | 0.01     | nan      | nan      | nan      | nan         | net1 (net)                            | net                       | net1                  |
# |  8 | nan      | nan      | 0.11     | 0.00     | 2.14     | ^           | _0_/A (sky130_fd_sc_hd__inv_2)        | sky130_fd_sc_hd__inv_2    | _0_/A                 |
# |  9 | nan      | nan      | 0.02     | 0.03     | 2.17     | v           | _0_/Y (sky130_fd_sc_hd__inv_2)        | sky130_fd_sc_hd__inv_2    | _0_/Y                 |
# | 10 | 1        | 0.00     | nan      | nan      | nan      | nan         | net2 (net)                            | net                       | net2                  |
# | 11 | nan      | nan      | 0.02     | 0.00     | 2.17     | v           | output2/A (sky130_fd_sc_hd__clkbuf_4) | sky130_fd_sc_hd__clkbuf_4 | output2/A             |
# | 12 | nan      | nan      | 0.08     | 0.18     | 2.36     | v           | output2/X (sky130_fd_sc_hd__clkbuf_4) | sky130_fd_sc_hd__clkbuf_4 | output2/X             |
# | 13 | 1        | 0.03     | nan      | nan      | nan      | nan         | out (net)                             | net                       | out                   |
# | 14 | nan      | nan      | 0.08     | 0.00     | 2.36     | v           | out (out)                             | out                       | out                   |
# | 15 | nan      | nan      | nan      | nan      | 2.36     | nan         | data arrival time                     | nan                       | nan                   |
# | 16 | nan      | nan      | 0.00     | 10.00    | 10.00    | nan         | clock __VIRTUAL_CLK__ (rise edge)     | rise edge                 | clock __VIRTUAL_CLK__ |
# | 17 | nan      | nan      | nan      | 0.00     | 10.00    | nan         | clock network delay (ideal)           | ideal                     | clock network delay   |
# | 18 | nan      | nan      | nan      | -0.25    | 9.75     | nan         | clock uncertainty                     | nan                       | nan                   |
# | 19 | nan      | nan      | nan      | 0.00     | 9.75     | nan         | clock reconvergence pessimism         | nan                       | nan                   |
# | 20 | nan      | nan      | nan      | -2.00    | 7.75     | nan         | output external delay                 | nan                       | nan                   |
# | 21 | nan      | nan      | nan      | nan      | 7.75     | nan         | data required time                    | nan                       | nan                   |
# | 22 | ------   | -------- | -------- | -------- | -------- | --          | ------------------------------------- | nan                       | nan                   |
# | 23 | nan      | nan      | nan      | nan      | 7.75     | nan         | data required time                    | nan                       | nan                   |
# | 24 | nan      | nan      | nan      | nan      | -2.36    | nan         | data arrival time                     | nan                       | nan                   |
#

# We can extract the propagation delay from the input and output frame accordingly.

piel.calculate_propagation_delay_from_file(file_path=run_output_sta_file_list[0])[0]

# |    |   index_x |   Fanout_out |   Cap_out |   Slew_out |   Delay_out |   Time_out | Direction_out   | Description_out   | net_type_out   | net_name_out   |   index_y |   Fanout_in |   Cap_in |   Slew_in |   Delay_in |   Time_in | Direction_in   | Description_in   | net_type_in   | net_name_in   |   propagation_delay |
# |---:|----------:|-------------:|----------:|-----------:|------------:|-----------:|:----------------|:------------------|:---------------|:---------------|----------:|------------:|---------:|----------:|-----------:|----------:|:---------------|:-----------------|:--------------|:--------------|--------------------:|
# |  0 |        24 |          nan |       nan |       0.08 |           0 |       2.36 | v               | out (out)         | out            | out            |        13 |         nan |      nan |      0.02 |       0.01 |      2.01 | ^              | in (in)          | in            | in            |                0.35 |
#
