"""
The objective of this file is to provide the simulation ports and interconnection to consider modelling digital and mixed signal logic.

The main simulation driver is cocotb, and this generates a set of files that correspond to time-domain digital
simulations. The cocotb verification software can also be used to perform mixed signal simulation, and digital data
can be inputted as a bitstream into a photonic solver, although the ideal situation would be to have integrated
photonic time-domain models alongside the electronic simulation solver, and maybe this is where it will go. It can be
assumed that, as is currently, cocotb can interface python with multiple solvers until someone (and I'd love to do
this) writes an equivalent python-based or C++ based python time-domain simulation solver.

The nice thing about cocotb is that as long as the photonic simulations can be written asynchronously, time-domain
simulations can be closely integrated or simulated through this verification software. """
import functools
import pathlib
import subprocess
from typing import Literal
from piel.file_system import return_path, write_script, delete_path_list_in_directory

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
    Checks if a cocotb testbench exists in the design directory.

    Args:
        design_directory(str | pathlib.Path): Design directory.

    Returns:
        cocotb_testbench_exists(bool): True if cocotb testbench exists.
    """
    cocotb_testbench_exists = False
    design_directory = return_path(design_directory)
    testbench_directory = design_directory / "tb"
    testbench_directory_exists = testbench_directory.exists()

    if testbench_directory_exists:
        # Check if cocotb python files are present
        cocotb_python_files = list(testbench_directory.glob("*.py"))
        if len(cocotb_python_files) > 0:
            cocotb_testbench_exists = True
        else:
            pass
    else:
        pass

    return cocotb_testbench_exists


def configure_cocotb_simulation(
    design_directory: str | pathlib.Path,
    simulator: Literal["icarus", "verilator"],
    top_level_language: Literal["verilog", "vhdl"],
    top_level_verilog_module: str,
    test_python_module: str,
    design_sources_list: list | None = None,
):
    """
    Writes a cocotb makefile.

    If no design_sources_list is provided then it adds all the design sources under the `src` folder.

    In the form
    .. code-block::

        #!/bin/sh
        # Makefile
        # defaults
        SIM ?= icarus
        TOPLEVEL_LANG ?= verilog

        # Note we need to include the test script to the PYTHONPATH
        export PYTHONPATH =

        VERILOG_SOURCES += $(PWD)/my_design.sv
        # use VHDL_SOURCES for VHDL files

        # TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
        TOPLEVEL := my_design

        # MODULE is the basename of the Python test file
        MODULE := test_my_design

        # include cocotb's make rules to take care of the simulator setup
        include $(shell cocotb-config --makefiles)/Makefile.sim


    Args:
        design_directory (str | pathlib.Path): The directory where the design is located.
        simulator (Literal["icarus", "verilator"]): The simulator to use.
        top_level_language (Literal["verilog", "vhdl"]): The top level language.
        top_level_verilog_module (str): The top level verilog module.
        test_python_module (str): The test python module.
        design_sources_list (list | None, optional): A list of design sources. Defaults to None.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    design_sources_directory = design_directory / "src"

    if design_sources_list is not None:
        # Include all the design sources files in a list
        design_sources_list = list(design_sources_directory.iterdir())

    top_commands_list = [
        "#!/bin/bash",
        "# Makefile",
        "SIM ?= " + simulator,
        "TOPLEVEL_LANG ?= " + top_level_language,
    ]

    middle_commands_list = []
    # TODO: Implement mixed source designs.
    if top_level_language == "verilog":
        for source_file in design_sources_list:
            middle_commands_list.append(
                "VERILOG_SOURCES += " + str(source_file.resolve())
            )
    elif top_level_language == "vhdl":
        for source_file in design_sources_list:
            middle_commands_list.append("VHDL_SOURCES += " + str(source_file.resolve()))

    bottom_commands_list = [
        "TOPLEVEL := " + top_level_verilog_module,
        "MODULE := " + test_python_module,
        "include $(shell cocotb-config --makefiles)/Makefile.sim",
    ]

    commands_list = []
    commands_list.extend(top_commands_list)
    commands_list.extend(middle_commands_list)
    commands_list.extend(bottom_commands_list)

    script = " \n".join(commands_list)
    write_script(
        directory_path=design_directory / "tb", script=script, script_name="Makefile"
    )


delete_simulation_output_files = functools.partial(
    delete_path_list_in_directory,
    path_list=["sim_build", "__pycache__", "ivl_vhdl_work"],
)


def run_cocotb_simulation(
    design_directory: str,
) -> subprocess.CompletedProcess:
    """
    Equivalent to running the cocotb makefile
    .. code-block::

        make

    Args:
        design_directory (str): The directory where the design is located.

    Returns:
        subprocess.CompletedProcess: The subprocess.CompletedProcess object.

    """
    test_directory = return_path(design_directory) / "tb"
    commands_list = ["cd " + str(test_directory.resolve()), "make"]
    script = "; \n".join(commands_list)
    # Save script if desired to run directly
    write_script(
        directory_path=test_directory,
        script=script,
        script_name="run_cocotb_simulation.sh",
    )
    run = subprocess.run(script, capture_output=True, shell=True, check=True)
    return run
