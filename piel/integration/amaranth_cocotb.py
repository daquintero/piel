from piel.types.digital import TruthTable
from ..file_system import return_path
from ..types import PathTypes


def create_cocotb_truth_table_verification_python_script(
    module: PathTypes,
    truth_table: TruthTable,
    test_python_module_name: str = "top_test",
):
    """
    Creates a cocotb test script for verifying logic defined by the truth table.

    Args:
        module (PathTypes): The path to the module where the test script will be placed.
        truth_table (TruthTable): A dictionary representing the truth table.
        test_python_module_name (str, optional): The name of the test python module. Defaults to "top_test".

    Example:
        truth_table = {
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1],
            "X": [0, 1, 1, 0]  # Expected output (for XOR logic, as an example)
        }
        create_cocotb_truth_table_verification_python_script(truth_table)
    """
    # Extract input and output ports
    input_ports = truth_table.input_ports
    output_ports = truth_table.output_ports

    # Get the implementation dictionary with only the specified ports
    truth_table_dict = truth_table.implementation_dictionary

    # Resolve the module path and create the tb directory if it doesn't exist
    module_path = return_path(module)
    tb_directory_path = module_path / "tb"
    tb_directory_path.mkdir(parents=True, exist_ok=True)
    python_module_test_file_path = tb_directory_path / f"{test_python_module_name}.py"
    output_file = tb_directory_path / "out" / "truth_table_test_results.csv"
    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure 'out' directory exists

    # Create the header for the script
    script_content = """
# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0
import cocotb
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time
import pandas as pd

@cocotb.test()
async def truth_table_test(dut):
    \"\"\"Test for logic defined by the truth table\"\"\"

"""
    # Extract signal names and values from the truth table
    signals = list(truth_table_dict.keys())
    num_tests = len(truth_table_dict[signals[0]])

    # Initialize lists to store signal files for logging
    for signal in signals:
        script_content += f"    {signal.lower()}_data = []\n"

    script_content += "    time = []\n\n"

    # Loop over each row in the truth table to generate test cases
    for i in range(num_tests):
        script_content += f"    # Test case {i + 1}\n"
        for signal in input_ports:  # Input ports are the inputs to the DUT
            value = truth_table_dict[signal][i]
            script_content += f'    dut.{signal}.value = cocotb.binary.BinaryValue("{value}")\n'  # Assign binary string values directly

        script_content += "    await Timer(2, units='ns')\n\n"

        # Check the expected output for each output port
        for output_signal in output_ports:
            expected_value = truth_table_dict[output_signal][i]
            script_content += f'    assert dut.{output_signal}.value == cocotb.binary.BinaryValue("{expected_value}"), '
            script_content += f'f"Test failed for inputs {input_ports}: expected {expected_value} but got {{dut.{output_signal}.value}}."\n'

        # Append files to lists for logging
        for signal in signals:
            script_content += f"    {signal.lower()}_data.append(dut.{signal}.value)\n"

        script_content += "    time.append(get_sim_time())\n\n"

    # Store the results in a CSV file
    script_content += "    simulation_data = {\n"
    for signal in signals:
        script_content += f'        "{signal.lower()}": {signal.lower()}_data,\n'
    script_content += '        "time": time\n'
    script_content += "    }\n\n"
    script_content += (
        f'    pd.DataFrame(simulation_data).to_csv("{str(output_file)}") \n'
    )

    # Write the script to a file
    with open(python_module_test_file_path, "w") as file:
        file.write(script_content)

    print(f"Test script written to {python_module_test_file_path}")
