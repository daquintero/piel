# # CocoTB Simulation

import piel

design_directory = "./designs" / piel.return_path("simple_design") / "simple_design"

# Check that there exist cocotb python test files already in our design directory:

piel.check_cocotb_testbench_exists(design_directory)

# Create a cocotb Makefile

piel.configure_cocotb_simulation(
    design_directory=design_directory,
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
# We first list the files for our design directory:
