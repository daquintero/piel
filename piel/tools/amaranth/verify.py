import amaranth as am
from amaranth.sim import Simulator, Delay
import types
from typing import Literal

from ...project_structure import get_module_folder_type_location
from ...file_system import return_path
from ...types import PathTypes

__all__ = ["verify_truth_table"]


def verify_truth_table(
    truth_table_amaranth_module: am.Elaboratable,
    truth_table_dictionary: dict,
    inputs: list,
    outputs: list,
    vcd_file_name: str,
    target_directory: PathTypes,
    implementation_type: Literal[
        "combinatorial", "sequential", "memory"
    ] = "combinatorial",
):
    """construct_amaranth_module_from_truth_table
    We will implement a function that tests the module to verify that the outputs generates match the truth table provided.

    TODO Implement a similar function from the openlane netlist too.
    TODO unclear they can implement verification without it being in a synchronous simulation.
    """

    def verify_logic():
        """
        This function implements the logic verification specifically.
        """
        input_port_signal = getattr(truth_table_amaranth_module, inputs[0]).eq
        output_port_signal = getattr(truth_table_amaranth_module, outputs[0])
        i = 0
        for input_value_i in truth_table_dictionary[inputs[0]]:
            # Iterate over the input value array and test every specific array.
            # Test against the output signal value in each clock cycle.
            yield input_port_signal(int(input_value_i))
            yield output_port_signal
            # Check that the output signal matches the design signal.
            # print(dir(output_port_signal))
            # print(output_port_signal.matches(0))
            # print(int(truth_table_dictionary[outputs[0]][i]))
            # assert output_port_signal == int(truth_table_dictionary[outputs[0]][i])
            yield Delay(1e-6)  # in a combinatorial simulation.
            i += 1

    # Determine the output data files directory
    if isinstance(target_directory, types.ModuleType):
        # If the path follows the structure of a `piel` path.
        target_directory = get_module_folder_type_location(
            module=target_directory, folder_type="digital_testbench"
        )
    else:
        # If not then just append the right path.
        target_directory = return_path(target_directory)

    # Set up synchronous simulation as I think it's necessary even for a combinatorial component as of 21/Ago/2023
    simulation = Simulator(truth_table_amaranth_module)
    simulation.add_process(verify_logic)

    if implementation_type == "synchronous":
        simulation.add_clock(1e-6)  # 1 MHz
        simulation.add_sync_process(truth_table_amaranth_module)
    elif implementation_type == "combinatorial":
        # TODO
        pass

    # Generate vcd outputs to verify?
    # TODO see how to generate a panads table from this accordingly.
    output_vcd_file = target_directory / vcd_file_name
    with simulation.write_vcd(str(output_vcd_file)):
        simulation.run()
    # TODO can we access inside here to generate a Pandas version.
