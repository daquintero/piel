from typing import Literal, Any

from ...project_structure import get_module_folder_type_location
from ...file_system import return_path
from ...types import PathTypes
from piel.types.digital import TruthTable

__all__ = ["verify_amaranth_truth_table"]


def verify_amaranth_truth_table(
    truth_table_amaranth_module: Any,
    truth_table: TruthTable,
    vcd_file_name: str,
    target_directory: PathTypes,
    implementation_type: Literal[
        "combinatorial", "sequential", "memory"
    ] = "combinatorial",
):
    """
    Verifies that the outputs generated by the given Amaranth module match the provided truth table.

    This function runs a simulation of the Amaranth module and checks if the outputs for each set of inputs
    match the expected outputs as specified in the truth table. It can optionally generate a VCD file for detailed analysis.

    Args:
        truth_table_amaranth_module (amaranth.Elaboratable): The Amaranth module to be verified.
        truth_table (TruthTable): The truth table specifying expected inputs and outputs.
        vcd_file_name (str): The name of the VCD file to generate for the simulation.
        target_directory (PathTypes): The directory where the VCD file will be saved. Can be a direct path or a module type path.
        implementation_type (Literal["combinatorial", "sequential", "memory"], optional):
            The type of implementation to simulate. Defaults to "combinatorial".

    Returns:
        None

    Raises:
        AttributeError: If the specified connection are not found in the Amaranth module.

    Examples:
        >>> am_module = MyAmaranthModule()  # Assuming this is a defined Amaranth module.
        >>> truth_table = TruthTable(
        >>>     input_ports=["input1"],
        >>>     output_ports=["output1", "output2"],
        >>>     input1=["0", "1"],
        >>>     output1=["1", "0"],
        >>>     output2=["0", "1"]
        >>> )
        >>> verify_amaranth_truth_table(am_module, truth_table, "output.vcd", "/path/to/save")
    """
    import amaranth as am
    from amaranth.sim import Simulator, Delay
    import types

    if isinstance(truth_table_amaranth_module, am.Elaboratable):
        pass
    else:
        raise AttributeError("Amaranth module should be am.Elaboratable")

    inputs = truth_table.input_ports
    outputs = truth_table.output_ports
    truth_table_df = truth_table.dataframe

    def verify_logic():
        """
        Implements the logic verification for the Amaranth module.

        This generator function sets the input signals and checks the output signals against
        the expected values from the truth table in each simulation cycle.
        """
        input_port_signal = getattr(truth_table_amaranth_module, inputs[0]).eq

        for i, input_value_i in enumerate(truth_table_df[inputs[0]]):
            # Apply input value
            yield input_port_signal(
                int(input_value_i, 2)
            )  # Convert input value to integer (assuming binary string)
            yield Delay(1e-6)  # Delay for combinatorial logic simulation

            # Check each output signal
            for output_port in outputs:
                output_port_signal = getattr(truth_table_amaranth_module, output_port)
                expected_output_value = int(truth_table_df[output_port].iloc[i], 2)
                assert (yield output_port_signal) == expected_output_value, (
                    f"Expected output {expected_output_value} on {output_port} for input {input_value_i} "
                    f"but got {(yield output_port_signal)}."
                )

    # Determine the output files files directory
    if isinstance(target_directory, types.ModuleType):
        target_directory = get_module_folder_type_location(
            module=target_directory, folder_type="digital_testbench"
        )
    else:
        target_directory = return_path(target_directory)

    # Ensure the target directory exists
    target_directory.mkdir(parents=True, exist_ok=True)

    output_vcd_file = target_directory / vcd_file_name

    # Set up the simulator for the Amaranth module
    simulation = Simulator(truth_table_amaranth_module)
    simulation.add_process(verify_logic)

    if implementation_type == "sequential":
        simulation.add_clock(1e-6)  # Add a clock for sequential logic
        simulation.add_sync_process(verify_logic)  # Sync process for sequential logic
    elif implementation_type == "combinatorial":
        # No clock is needed for combinatorial logic
        pass
    elif implementation_type == "memory":
        # Add specific handling for memory-based implementations if needed
        pass

    # Run the simulation and write VCD output for verification
    with simulation.write_vcd(str(output_vcd_file)):
        simulation.run()

    print(f"VCD file generated and written to {output_vcd_file}")
