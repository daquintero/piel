import amaranth as am
from amaranth.sim import Simulator
from typing import Literal

import piel
from ...config import piel_path_types

__all__ = ["verify_truth_table"]


def verify_truth_table(
    truth_table_amaranth_module: am.Elaboratable,
    truth_table_dictionary: dict,
    inputs: list,
    outputs: list,
    vcd_file_name: str,
    target_output_directory: piel_path_types,
    implementation_type: Literal[
        "combinatorial", "sequential", "memory"
    ] = "combinatorial",
):
    """
    We will implement a function that tests the module to verify that the outputs generates match the truth table provided.

    TODO Implement a similar function from the openlane netlist too.
    TODO unclear they can implement verification without it being in a synchronous simulation.
    """

    def verify_logic():
        """
        This function implements the logic verification specifically.
        """
        input_port_signal = getattr(truth_table_amaranth_module, inputs[0]).eq
        output_port_signal = getattr(truth_table_amaranth_module, outputs[0]).eq
        i = 0
        for input_value_i in truth_table_dictionary[inputs[0]]:
            # Iterate over the input value array and test every specific array.
            # Test against the output signal value.
            yield input_port_signal(input_value_i)
            # Check that the output signal matches the design signal.
            assert output_port_signal == int(truth_table_dictionary[outputs[0]][i])
            i += 1

    target_output_directory = piel.return_path(target_output_directory)
    verify_logic()
    simulation = Simulator(truth_table_amaranth_module)

    if implementation_type == "synchronous":
        simulation.add_clock(1e-6)  # 1 MHz
        simulation.add_sync_process(truth_table_amaranth_module)

    # Generate vcd outputs to verify?
    # TODO see how to generate a panads table from this accordingly.
    output_vcd_file = target_output_directory / vcd_file_name
    with simulation.write_vcd(str(output_vcd_file)):
        simulation.run()
    # TODO can we access inside here to generate a Pandas version.
