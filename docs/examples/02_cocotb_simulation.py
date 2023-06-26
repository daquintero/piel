# # CocoTB Simulation

import pathlib
import piel

design_directory = pathlib.Path("simple_design")

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

# Now we can create the simulation output files from the `makefile`

# Run cocotb simulation
piel.run_cocotb_simulation(design_directory)
