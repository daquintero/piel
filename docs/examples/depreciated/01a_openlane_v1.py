# ## OpenLane V1 Flow - Unsupported/Depreciated

# ### Interacting with Existing Designs
# Let us first begin by exploring the default `spm` design. Note that this design has to be included in the `OPENLANE_ROOT/<latestversion>/designs` directory for this to be valid, or the `root_directory` parameter has to be setup depending on your environment.

import piel

# Configure our example design name:
design_name = "spm"

# Check that the design exists in the OpenLane v1 design folder:
piel.check_design_exists_openlane_v1(design_name)

# Check if `config.json` has already been provided for this design.
piel.check_config_json_exists_openlane_v1(design_name)

# Read the `config.json` file for this design:
example_config_json = piel.read_configuration_openlane_v1(design_name)
example_config_json

# Get the design directory for this design:
design_directory = piel.get_design_directory_from_root_openlane_v1(design_name)
design_directory

# ### Run Flow `spm` Design using `piel`

# #### The Fast Version

# `piel` provides a set of functions for easily configuring and running a design into `Openlane v1`. For the default `spm` that already has a set up `config.json` file and project structure inside `$OPENLANE_ROOT/<latestversion>/designs`:


piel.configure_and_run_design_openlane_v1(
    design_name=design_name,
)

# #### The Slow Version


# Get the latest version OpenLane v1 root directory:

root_directory = piel.get_latest_version_root_openlane_v1()
root_directory

# Check that the design_directory provided is under $OPENLANE_ROOT/<"latestversion">/designs:

design_exists = piel.check_design_exists_openlane_v1(design_name)
design_directory = root_directory / "designs" / design_name
design_exists

# Check if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`

config_json_exists = piel.check_config_json_exists_openlane_v1(design_name)
config_json_exists

# Create a script directory, a script is written and permissions are provided for it to be executable.

piel.configure_flow_script_openlane_v1(design_name=design_name)

# Permit and execute the `openlane_flow.sh` script in the `scripts` directory.

openlane_flow_script_path = design_directory / "scripts" / "openlane_flow.sh"
piel.permit_script_execution(openlane_flow_script_path)
piel.run_script(openlane_flow_script_path)

# ### Creating Parametric Designs

# Let us first copy our `simple_design` into the root design directory. Note that as in most OPENLANE environments, operations are performed with root access to a certain level and in this environment you need to do this with root access.

openlane_v1_designs_directory = piel.get_latest_version_root_openlane_v1() / "designs"

example_design_name = "simple_design"

piel.copy_source_folder(
    source_directory="./designs" / piel.return_path(example_design_name),
    target_directory=openlane_v1_designs_directory,
)

# Now we check whether the simple_design has been copied.
piel.check_design_exists_openlane_v1(example_design_name)

# # This will now become our base design.
base_design_directory = piel.get_design_directory_from_root_openlane_v1(
    design_name=example_design_name
)
base_design_directory

# We can read in the default OpenLane v1 configuration file:
base_configuration = piel.read_configuration_openlane_v1(
    design_name=example_design_name
)
base_configuration

# #### Configure Parametric Parameter
# We will create a parametric parameter that exists within the `configuration` dictionary. One very common and useful one in `OpenLane` is one of the `yosys` `SYNTH_PARAMETERS` which changes the synthesis parameters of the system. See `docs/section/parametric` for more details. We can easily create a "multi-parameter sweep" with some built-in `piel` utilities:

example_verilog_parameter_iteration = list()
for parameter_value in [2, 4, 8, 16, 32, 64, 128]:
    example_verilog_parameter_iteration.append(
        "MY_PARAMETER_VALUE=" + str(int(parameter_value))
    )

synthesis_parameter_iteration = {
    "SYNTH_PARAMETERS": example_verilog_parameter_iteration
}

# `piel` provides some useful functions to check multi-parameter sweeps:

parametrised_configuration_list = piel.configure_parametric_designs_openlane_v1(
    design_name=example_design_name,
    parameter_sweep_dictionary=synthesis_parameter_iteration,
)

# This function allows you to check the parameter configuration before creating the parametrised directories. Once you are ready you can do with the same `parameter_sweep_dictionary`:

piel.create_parametric_designs_openlane_v1(
    design_name=example_design_name,
    parameter_sweep_dictionary=synthesis_parameter_iteration,
)

# You should be able to verify these designs have been set correctly in the root directory:

# ### GDSFactory-OpenLane Layout Integration

# This is the simplest implementation of the integration between `OpenLane` and `gdsfactory`.

# We can find which is the latest design run

latest_design_run_directory = piel.find_latest_design_run(
    design_directory="./designs" / piel.return_path("inverter"),
)
latest_design_run_directory


# We can test this with a provided pre-build OpenLane design.

inverter_component = piel.create_gdsfactory_component_from_openlane(
    design_directory="./designs" / piel.return_path("inverter"),
)

inverter_component.plot_widget()

# ![inverter_component_plot_widget](../_static/img/examples/01b_gdsfactory_layout_integation/inverter_component_plot_widget.PNG)

# ### OpenLane Output Analysis

# First, we get the directory of the latest run:

latest_run_output = piel.find_latest_design_run(
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

# ```python
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

# Say we want to explore the output of one particular timing file. We can extract all the timing files accordingly:

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
# | 15 | nan      | nan      | nan      | nan      | 2.36     | nan         | files arrival time                     | nan                       | nan                   |
# | 16 | nan      | nan      | 0.00     | 10.00    | 10.00    | nan         | clock __VIRTUAL_CLK__ (rise edge)     | rise edge                 | clock __VIRTUAL_CLK__ |
# | 17 | nan      | nan      | nan      | 0.00     | 10.00    | nan         | clock network delay (ideal)           | ideal                     | clock network delay   |
# | 18 | nan      | nan      | nan      | -0.25    | 9.75     | nan         | clock uncertainty                     | nan                       | nan                   |
# | 19 | nan      | nan      | nan      | 0.00     | 9.75     | nan         | clock reconvergence pessimism         | nan                       | nan                   |
# | 20 | nan      | nan      | nan      | -2.00    | 7.75     | nan         | output external delay                 | nan                       | nan                   |
# | 21 | nan      | nan      | nan      | nan      | 7.75     | nan         | files required time                    | nan                       | nan                   |
# | 22 | ------   | -------- | -------- | -------- | -------- | --          | ------------------------------------- | nan                       | nan                   |
# | 23 | nan      | nan      | nan      | nan      | 7.75     | nan         | files required time                    | nan                       | nan                   |
# | 24 | nan      | nan      | nan      | nan      | -2.36    | nan         | files arrival time                     | nan                       | nan                   |
#

# We can extract the propagation delay from the input and output frame accordingly.

piel.calculate_propagation_delay_from_file(file_path=run_output_sta_file_list[0])[0]

# |    |   index_x |   Fanout_out |   Cap_out |   Slew_out |   Delay_out |   Time_out | Direction_out   | Description_out   | net_type_out   | net_name_out   |   index_y |   Fanout_in |   Cap_in |   Slew_in |   Delay_in |   Time_in | Direction_in   | Description_in   | net_type_in   | net_name_in   |   propagation_delay |
# |---:|----------:|-------------:|----------:|-----------:|------------:|-----------:|:----------------|:------------------|:---------------|:---------------|----------:|------------:|---------:|----------:|-----------:|----------:|:---------------|:-----------------|:--------------|:--------------|--------------------:|
# |  0 |        24 |          nan |       nan |       0.08 |           0 |       2.36 | v               | out (out)         | out            | out            |        13 |         nan |      nan |      0.02 |       0.01 |      2.01 | ^              | in (in)          | in            | in            |                0.35 |
#

# ### Interacting with the `piel` flow

# `piel` provides an integrated toolset for codesigning and tapeing out analog, digital and photonic chips in a single design flow that is extendable to probably any Python-binded software you might deire. There is a recommended project structure in order to have a clear structure of the location of the files without conflicts between different toolsets. This is described in the TODO ADD PROJECT STRUCTURE DOCUMENTATION LINK.
#
# Each project can be interacted as a python module, and different sections of the system design can be accessed accordingly. `piel` provides some examples of this in `docs/examples/designs/`. These project design flows are further demonstrated and derived in the next examples. For now, let's take some digital verilog source files generated in the `amaranth_driven_flow` design, and configure it to layout an `OpenLane v1` chip from it. Run this from your terminal:
#
# ```shell
# # cd piel/docs/examples/designs/amaranth_driven_flow
# pip install -e .
# ```

import amaranth_driven_flow

# Let's find the directory path of our module:

piel.return_path(amaranth_driven_flow)

# ```python
# WindowsPath('c:/users/dario/documents/phd/piel/docs/examples/designs/amaranth_driven_flow/amaranth_driven_flow/__init__.py/..')
# ```

# We can get an example structure of an `openlane` configuration dictionary that is compatible for `amaranth` generated logic, as specific naming conventions need to be followed to generate the outputs.

our_amaranth_openlane_config = (
    piel.tools.openlane.defaults.test_basic_open_lane_configuration
)
our_amaranth_openlane_config

piel.write_configuration_openlane_v1(
    configuration=our_amaranth_openlane_config,
    design_directory=amaranth_driven_flow,
)

# Let's first copy this design into the `Openlane v1` root directory so we can run the flow as normal:


piel.copy_source_folder(
    source_directory=amaranth_driven_flow,
    target_directory=openlane_v1_designs_directory,
)

# We can read it has been written properly easily as described before:

base_configuration = piel.read_configuration_openlane_v1(
    design_name=amaranth_driven_flow.__name__
)
