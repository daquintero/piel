# # Digital Design & Simulation Flow

# There are many tools to perform a digital design flow. I have summarised some of the most relevant ones in the TODO LINK INTEGRATION PAGE. In `piel`, there are a few digital design flow functionalities integrated. However, because the output or interconnections have a set of accepted common types, it is possible to use different design flows to integrate with other tools in the flow.
#
# We will explore two popular ones:
# * `amaranth` aims to be more than just Python-to-HDL design, but a full digital-flow package design tool.
# * `cocotb` is mainly used for writing testbenches in Python and verification of logic.

import piel
import simple_design

# In this example, we will use `amaranth` to perform some design and then simulations, so let's create a suitable project structure based on our initial `simple_design`, where we will output our files.
# However, note that the `amaranth<0.4` project has of the 23/Ago/2023 a versioning problem which means that they haven't released a new version for the last two years and this is conflicting with other packages. As such, you have to install the latest `amaranth` on your own. However, when you do, you can then run this:

from piel.tools.amaranth import (
    construct_amaranth_module_from_truth_table,
    generate_verilog_from_amaranth,
    verify_truth_table,
)

# +
# Uncomment this if you want to run it for the first time.
# piel.create_empty_piel_project(
#     project_name="amaranth_driven_flow", parent_directory="../designs/"
# )
# -

# We can also automate the `pip` installation of our local module:

# +
#  Uncomment this if you want to run it for the first time.
# piel.pip_install_local_module("../designs/amaranth_driven_flow")
# -

# We can check that this has been installed. You might need to restart your `jupyter` kernel.

import amaranth_driven_flow

amaranth_driven_flow

# ```python
# <module 'amaranth_driven_flow' from 'c:\\users\\dario\\documents\\phd\\piel\\docs\\examples\\designs\\amaranth_driven_flow\\amaranth_driven_flow\\__init__.py'>
# ```

# ## `amaranth` Design Flow

# Let's design a basic digital module from their example which we can use later. `amaranth` is one of the great Python digital design flows. I encourage you to check out their documentation for basic examples. We will explore a particular `codesign` use case: creating bespoke logic for detector optical signals.
#
# Also, another aspect is that `amaranth` follow a very declarative class-based design flow, which can be tricky to connect with other tools, so `piel` provides some construction functions that are more easily integrated in the current design flow.
#
# Say we have a table of input optical detector click signals that we want to route some phases to map to some outputs and we want to create a mapping. For now, let's consider a basic arbitrary truth table in this form:
#
# |Input Detector Signals | Output Phase Map|
# |-----------------------|-----------------|
# | 00 | 00 |
# | 01 | 10 |
# | 10 | 11 |
# | 11 | 11 |
#
# `piel` provides some easy functions to perform this convertibility. Say, we provide this information as a dictionary where the keys are the names of our input and output signals. This is a similar principle if you have a detector-triggered DAC bit configuration too.

detector_phase_truth_table = {
    "detector_in": ["00", "01", "10", "11"],
    "phase_map_out": ["00", "10", "11", "11"],
}


input_ports_list = ["detector_in"]
output_ports_list = ["phase_map_out"]
our_truth_table_module = construct_amaranth_module_from_truth_table(
    truth_table=detector_phase_truth_table,
    inputs=input_ports_list,
    outputs=output_ports_list,
)

# `amaranth` is much easier to use than other design flows like `cocotb` because it can be purely interacted with in `Python`, which means there are fewer complexities of integration. However, if you desire to use this with other digital layout tools, for example, `OpenROAD` as we have previously seen and maintain a coherent project structure with the photonics design flow, `piel` provides some helper functions to achieve this easily.
#
# We can save this file directly into our working examples directory.

ports_list = input_ports_list + output_ports_list
generate_verilog_from_amaranth(
    amaranth_module=our_truth_table_module,
    ports_list=ports_list,
    target_file_name="our_truth_table_module.v",
    target_directory=".",
)

# Another aspect is that as part of the `piel` flow, we have thoroughly thought of how to structure a codesign electronic-photonic project in order to be able to utilise all the range of tools in the process. You might want to save your design and simulation files to their corresponding locations so you can reuse them with another toolset in the future.
#
# Say, you want to append them to the `amaranth_driven_flow` project:

design_directory = piel.return_path(amaranth_driven_flow)

# Some functions you might want to use to save the designs in these directories are:

amaranth_driven_flow_src_folder = piel.get_module_folder_type_location(
    module=amaranth_driven_flow, folder_type="digital_source"
)

ports_list = input_ports_list + output_ports_list
generate_verilog_from_amaranth(
    amaranth_module=our_truth_table_module,
    ports_list=ports_list,
    target_file_name="our_truth_table_module.v",
    target_directory=amaranth_driven_flow_src_folder,
)

# Another thing we can do is verify that our implemented logic is valid. Creating a simulation is also useful in the future when we simulate our extracted place-and-route netlist in relation to the expected applied logic.

verify_truth_table(
    truth_table_amaranth_module=our_truth_table_module,
    truth_table_dictionary=detector_phase_truth_table,
    inputs=input_ports_list,
    outputs=output_ports_list,
    vcd_file_name="our_truth_table_module.vcd",
    target_directory=".",
)

# You can also use the module directory to automatically save the testbench in these functions.

verify_truth_table(
    truth_table_amaranth_module=our_truth_table_module,
    truth_table_dictionary=detector_phase_truth_table,
    inputs=input_ports_list,
    outputs=output_ports_list,
    vcd_file_name="our_truth_table_module.vcd",
    target_directory=amaranth_driven_flow,
)

# You can observe the design directory of the provided `amaranth_driven_flow` folder to verify that the files have been included in the other flow.
#
# We can see that the truth table logic has been accurately implemented in the post `vcd` verification test output generated.

# ![example_truth_table_verification](../../_static/img/examples/02_cocotb_simulation/example_truth_table_verification.PNG)

# ## `cocoTb` Simulation

# It is strongly encouraged to get familiar with the `piel` flow project structure, as this file directory distribution enables the easy use between multiple design tools without conflicts or without structured organisation.

# Location of our output files

source_output_files_directory = (
    piel.get_module_folder_type_location(
        module=simple_design, folder_type="digital_source"
    )
    / "out"
)
simulation_output_files_directory = (
    piel.get_module_folder_type_location(
        module=simple_design, folder_type="digital_testbench"
    )
    / "out"
)
simulation_output_files_directory

simulation_output_files_directory.exists()

# Check that there exist cocotb python test files already in our design directory:

piel.check_cocotb_testbench_exists(simple_design)

# Create a cocotb Makefile

piel.configure_cocotb_simulation(
    design_directory=simple_design,
    simulator="icarus",
    top_level_language="verilog",
    top_level_verilog_module="adder",
    test_python_module="test_adder",
    design_sources_list=list((design_directory / "src").iterdir()),
)

# Now we can create the simulation output files from the `makefile`. Note this will only work in our configured Linux environment.

# Run cocotb simulation
piel.run_cocotb_simulation(design_directory)

# However, what we would like to do is extract timing information of the circuit in Python and get corresponding
# graphs. We would like to have this digital signal information interact with our photonics model. Note that when
# running a `cocotb` simulation, this is done through asynchronous coroutines, so it is within the testbench file
# that any interaction and modelling with the photonics networks is implemented.

# ## Data Visualisation

# The user is free to write their own data visualisation structure and toolset. Conveniently, `piel` does provide a standard plotting tool for these type of `cocotb` signal outputs accordingly.
#
# We first list the files for our design directory, which is out `simple_design` local package if you have followed the correct installation instructions, and we can input our module import rather than an actual path unless you desire to customise.

cocotb_simulation_output_files = piel.get_simulation_output_files_from_design(
    simple_design
)
cocotb_simulation_output_files

# ```python
# ['C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\simple_design\\simple_design\\tb\\out\\adder_randomised_test.csv']
# ```

# We can read the simulation output data accordingly:

example_simple_simulation_data = piel.read_simulation_data(
    cocotb_simulation_output_files[0]
)
example_simple_simulation_data

# Now we can plot the corresponding data using the built-in interactive `bokeh` signal analyser function:

piel.simple_plot_simulation_data(example_simple_simulation_data)

# This looks like this:

# ![example_simple_design_outputs](../../_static/img/examples/02_cocotb_simulation/example_simple_design_outputs.PNG)
