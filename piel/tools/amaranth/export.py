import amaranth as am
from amaranth.back import verilog
import types
from ...config import piel_path_types

__all__ = ["generate_verilog_from_amaranth"]


def generate_verilog_from_amaranth(
    amaranth_module: am.Elaboratable,
    ports_list: list[str],
    target_file_name: str,
    target_directory: piel_path_types,
    backend=verilog,
) -> None:
    """
    This function exports an amaranth module to either a defined path, or a project structure in the form of an
    imported multi-design module.

    Iterate over ports list and construct a list of references for the strings provided in ``ports_list``

    TODO DOCS parameters.

    """
    if isinstance(target_directory, types.ModuleType):
        # If the path follows the structure of a `piel` path.
        target_file_path = target_directory / "src" / target_file_name
    else:
        # If not then just append the right path.
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
