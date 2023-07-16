# # CocoTB Simulation

import piel
import simple_design

# Location of our output files

design_directory = piel.return_path(simple_design)
simulation_output_files_directory = piel.return_path(simple_design) / "tb" / "out"
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

# ![example_simple_design_outputs](../_static/img/examples/02_cocotb_simulation/example_simple_design_outputs.PNG)
