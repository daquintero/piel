"""
This module provides a utility to generate Verilog code from an Amaranth module based on a given truth table.
It handles the conversion and export process, integrating with a specified file system structure.
"""

from typing import Any, Literal
from ...file_system import return_path
from ...project_structure import get_module_folder_type_location
from ...types import PathTypes, TruthTable


def generate_verilog_from_amaranth_truth_table(
    amaranth_module: Any,
    truth_table: TruthTable,
    target_file_name: str,
    target_directory: PathTypes,
    backend: Literal["verilog", "vhdl"] = "verilog",
) -> None:
    """
    Exports an Amaranth module to Verilog code and writes it to a specified path.

    This function converts an Amaranth elaboratable class to Verilog using the specified backend
    and writes the generated code to a file in the target directory. It supports both direct paths
    and paths defined by the project's module structure.

    Args:
        amaranth_module (amaranth.Elaboratable): The Amaranth module to be converted.
        truth_table (TruthTable): A truth table object containing input and output connection.
        target_file_name (str): The name of the target file to write the Verilog code to.
        target_directory (PathTypes): The target directory where the file will be saved.
            Can be a direct path or a module type path.
        backend (amaranth.back.verilog, optional): The backend to use for Verilog conversion. Defaults to `verilog`.

    Returns:
        None

    Raises:
        AttributeError: If any port specified in the truth table is not found in the Amaranth module.

    Examples:
        >>> am_module = MyAmaranthModule()  # Assuming this is a defined Amaranth module.
        >>> truth_table = TruthTable(
        >>>     input_ports=["input1", "input2"],
        >>>     output_ports=["output1"],
        >>>     input1=["00", "01", "10", "11"],
        >>>     output1=["0", "1", "1", "0"]
        >>> )
        >>> generate_verilog_from_amaranth_truth_table(am_module, truth_table, "output.v", "/path/to/save")
    """
    import amaranth as am
    import types

    if backend == "verilog":
        from amaranth.back import verilog

        backend = verilog
    elif backend == "vhdl":
        raise NotImplementedError("This backend is not yet implemented.")

    if isinstance(amaranth_module, am.Elaboratable):
        pass
    else:
        raise AttributeError("Amaranth module should be am.Elaboratable")

    ports_list = truth_table.ports_list

    if isinstance(target_directory, types.ModuleType):
        # If the path follows the structure of a `piel` path, get the appropriate folder type location.
        target_directory = get_module_folder_type_location(
            module=target_directory, folder_type="digital_source"
        )
    else:
        # Otherwise, convert the target directory to a path.
        target_directory = return_path(target_directory)

    target_file_path = target_directory / target_file_name

    # Iterate over connection list and construct a list of references for the strings provided in `keys_list`.
    module_ports_list = []
    for port_i in ports_list:
        if hasattr(amaranth_module, port_i):
            module_port_i = getattr(amaranth_module, port_i)
            module_ports_list.append(module_port_i)
        else:
            raise AttributeError(f"Port {port_i} not found in the Amaranth module.")

    # Ensure the directory exists.
    target_directory.mkdir(parents=True, exist_ok=True)

    # Write the Verilog file to the target path.
    with open(target_file_path, "w") as file:
        file.write(
            backend.convert(
                amaranth_module,
                ports=module_ports_list,
            )
        )

    print(f"Verilog file generated and written to {target_file_path}")
