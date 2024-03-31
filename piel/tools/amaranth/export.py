import amaranth as am
from amaranth.back import verilog
import types

from ...project_structure import get_module_folder_type_location
from ...file_system import return_path
from ...types import PathTypes

__all__ = ["generate_verilog_from_amaranth"]


def generate_verilog_from_amaranth(
    amaranth_module: am.Elaboratable,
    ports_list: list[str],
    target_file_name: str,
    target_directory: PathTypes,
    backend=verilog,
) -> None:
    """
    This function exports an amaranth module to either a defined path, or a project structure in the form of an
    imported multi-design module.

    Iterate over ports list and construct a list of references for the strings provided in ``ports_list``

    Args:
        amaranth_module (amaranth.Elaboratable): Amaranth elaboratable class.
        ports_list (list[str]): List of input names.
        target_file_name (str): Target file name.
        target_directory (PathTypes): Target directory PATH.
        backend (amaranth.back.verilog): Backend to use. Defaults to ``verilog``.

    Returns:
        None
    """
    if isinstance(target_directory, types.ModuleType):
        # If the path follows the structure of a `piel` path.
        target_directory = get_module_folder_type_location(
            module=target_directory, folder_type="digital_source"
        )
    else:
        # If not then just append the right path.
        target_directory = return_path(target_directory)
    target_file_path = target_directory / target_file_name

    # Iterate over ports list and construct a list of references for the strings provided in `ports_list`
    # TODO maybe compose this as a separate function.
    module_ports_list = list()
    for port_i in ports_list:
        module_port_i = getattr(amaranth_module, port_i)
        module_ports_list.append(module_port_i)

    with open(target_file_path, "w") as file:
        file.write(
            backend.convert(
                amaranth_module,
                ports=module_ports_list,
            )
        )
