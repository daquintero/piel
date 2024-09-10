"""
The objective of this file is to provide the simulation ports and interconnection to consider modelling digital and mixed signal logic.

The main simulation driver is cocotb, and this generates a set of files that correspond to time-domain digital
simulations. The cocotb verification software can also be used to perform mixed signal simulation, and digital files
can be inputted as a bitstream into a photonic solver, although the ideal situation would be to have integrated
photonic time-domain measurement alongside the electronic simulation solver, and maybe this is where it will go. It can be
assumed that, as is currently, cocotb can interface python with multiple solvers until someone (and I'd love to do
this) writes an equivalent python-based or C++ based python time-domain simulation solver.

The nice thing about cocotb is that as long as the photonic simulations can be written asynchronously, time-domain
simulations can be closely integrated or simulated through this verification software."""

import functools
import pathlib
import subprocess
from piel.file_system import return_path, write_file, delete_path_list_in_directory
from piel.types.digital import HDLSimulator, HDLTopLevelLanguage

__all__ = [
    "check_cocotb_testbench_exists",
    "configure_cocotb_simulation",
    "delete_simulation_output_files",
    "run_cocotb_simulation",
]


def check_cocotb_testbench_exists(
    design_directory: str | pathlib.Path,
) -> bool:
    """
    Checks if a Cocotb testbench exists in the specified design directory.

    Args:
        design_directory (str | pathlib.Path): The directory where the design files are located.

    Returns:
        bool: True if a Cocotb testbench exists, False otherwise.

    Examples:
        >>> check_cocotb_testbench_exists("/path/to/design")
        True
    """
    design_directory = return_path(design_directory)
    testbench_directory = design_directory / "tb"
    testbench_directory_exists = testbench_directory.exists()

    if testbench_directory_exists:
        # Check if there are Python files in the testbench directory excluding __init__.py
        cocotb_python_files = list(testbench_directory.glob("*.py"))
        if len(cocotb_python_files) > 1:
            return True
    return False


def configure_cocotb_simulation(
    design_directory: str | pathlib.Path,
    simulator: HDLSimulator,
    top_level_language: HDLTopLevelLanguage,
    top_level_verilog_module: str,
    test_python_module: str,
    design_sources_list: list | None = None,
) -> pathlib.Path:
    """
    Configures a Cocotb simulation by generating a Makefile in the specified directory.

    This function creates a Makefile required to run Cocotb simulations. It includes paths to design source files and sets up the simulator and language options.

    Args:
        design_directory (str | pathlib.Path): The directory where the design files are located.
        simulator (Literal["icarus", "verilator"]): The simulator to use for the simulation.
        top_level_language (Literal["verilog", "vhdl"]): The top-level HDL language used in the design.
        top_level_verilog_module (str): The top-level Verilog module name.
        test_python_module (str): The Python test module name for Cocotb.
        design_sources_list (list | None, optional): A list of design source file paths. Defaults to None.

    Returns:
        pathlib.Path: The path to the generated Makefile.

    Examples:
        >>> configure_cocotb_simulation("/path/to/design", "icarus", "verilog", "top_module", "test_module")
        PosixPath('/path/to/design/tb/Makefile')
    """
    design_directory = return_path(design_directory)
    design_sources_directory = design_directory / "src"

    if design_sources_list is None:
        # Include all design source files under the `src` folder if none are provided.
        design_sources_list = list(design_sources_directory.iterdir())

    # Top-level commands for the Makefile
    top_commands_list = [
        "#!/bin/bash",
        "# Makefile",
        "SIM ?= " + simulator,
        "TOPLEVEL_LANG ?= " + top_level_language,
    ]

    # Middle section commands to include design source files
    middle_commands_list = []
    if top_level_language == "verilog":
        for source_file in design_sources_list:
            middle_commands_list.append(
                "VERILOG_SOURCES += " + str(source_file.resolve())
            )
    elif top_level_language == "vhdl":
        for source_file in design_sources_list:
            middle_commands_list.append("VHDL_SOURCES += " + str(source_file.resolve()))

    # Bottom section commands to set top-level module and include Cocotb Makefile rules
    bottom_commands_list = [
        "TOPLEVEL := " + top_level_verilog_module,
        "MODULE := " + test_python_module,
        "include $(shell cocotb-config --makefiles)/Makefile.sim",
    ]

    # Combine all command lists into a single script
    commands_list = []
    commands_list.extend(top_commands_list)
    commands_list.extend(middle_commands_list)
    commands_list.extend(bottom_commands_list)

    script = " \n".join(commands_list)
    makefile_path = design_directory / "tb" / "Makefile"
    write_file(
        directory_path=design_directory / "tb", file_text=script, file_name="Makefile"
    )

    print(script)
    return makefile_path


# Partial function to delete specific simulation output files from a directory
delete_simulation_output_files = functools.partial(
    delete_path_list_in_directory,
    path_list=["sim_build", "__pycache__", "ivl_vhdl_work"],
)


def run_cocotb_simulation(
    design_directory: str,
) -> subprocess.CompletedProcess:
    """
    Runs the Cocotb simulation by executing the Makefile in the specified design directory.

    Args:
        design_directory (str): The directory where the design files are located.

    Returns:
        subprocess.CompletedProcess: The completed process object containing the result of the simulation run.

    Examples:
        >>> run_cocotb_simulation("/path/to/design")
    """
    test_directory = return_path(design_directory) / "tb"
    commands_list = ["cd " + str(test_directory.resolve()), "make"]
    script = "; \n".join(commands_list)

    # Save the script to a file for potential direct execution
    write_file(
        directory_path=test_directory,
        file_text=script,
        file_name="run_cocotb_simulation.sh",
    )

    try:
        # Execute the script and capture the output
        run = subprocess.run(script, capture_output=True, shell=True, check=True)

        # Print the standard output and standard error
        print("Standard Output (stdout):")
        print(run.stdout.decode())  # Decode bytes to string
        print("Standard Error (stderr):")
        print(run.stderr.decode())  # Decode bytes to string

        return run

    except subprocess.CalledProcessError as e:
        # Print detailed error information
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print("Standard Output (stdout):")
        print(e.stdout.decode())  # Decode bytes to string
        print("Standard Error (stderr):")
        print(e.stderr.decode())  # Decode bytes to string

        raise
