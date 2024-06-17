import pytest
import amaranth as am
from amaranth.sim import Simulator, Settle
from piel.tools.amaranth import (
    construct_amaranth_module_from_truth_table,
)  # Adjust the import based on your actual module structure
from piel.types import TruthTable

# TODO fix this
# # Helper function to run a simulation
# def simulate_module(module, inputs, input_values, expected_outputs):
#     sim = Simulator(module)
#
#     def process():
#         for input_value, expected_output in zip(input_values, expected_outputs):
#             yield getattr(module, inputs[0]).eq(int(input_value, 2))
#             yield Settle()
#             for output_name, expected_value in expected_output.items():
#                 output_signal = getattr(module, output_name)
#                 assert (yield output_signal) == int(expected_value, 2)
#             yield
#
#     sim.add_process(process)
#     sim.run()

# Tests for construct_amaranth_module_from_truth_table function
# def test_combinatorial_truth_table():
#     truth_table_data = {
#         "input_port": ["00", "01", "10", "11"],
#         "output_port": ["00", "10", "11", "01"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input_port"],
#         output_ports=["output_port"],
#         **truth_table_data
#     )
#
#     am_module = construct_amaranth_module_from_truth_table(truth_table, implementation_type="combinatorial")
#
#     assert isinstance(am_module, am.Elaboratable)
#
#     # Simulate the module to verify its behavior
#     simulate_module(
#         am_module,
#         truth_table.input_ports,
#         truth_table_data["input_port"],
#         [{"output_port": v} for v in truth_table_data["output_port"]]
#     )


def test_empty_truth_table():
    truth_table_data = {"input_port": [], "output_port": []}
    truth_table = TruthTable(
        input_ports=["input_port"], output_ports=["output_port"], **truth_table_data
    )

    with pytest.raises(ValueError):
        construct_amaranth_module_from_truth_table(
            truth_table, logic_implementation_type="combinatorial"
        )


# TODO fix this
# def test_multiple_outputs():
#     truth_table_data = {
#         "input_port": ["00", "01", "10", "11"],
#         "output_port1": ["00", "10", "11", "01"],
#         "output_port2": ["11", "01", "10", "00"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input_port"],
#         output_ports=["output_port1", "output_port2"],
#         **truth_table_data
#     )
#
#     am_module = construct_amaranth_module_from_truth_table(truth_table, implementation_type="combinatorial")
#
#     assert isinstance(am_module, am.Elaboratable)
#
#     # Simulate the module to verify its behavior with multiple outputs
#     simulate_module(
#         am_module,
#         truth_table.input_ports,
#         truth_table_data["input_port"],
#         [{"output_port1": v1, "output_port2": v2} for v1, v2 in zip(truth_table_data["output_port1"], truth_table_data["output_port2"])]
#     )


def test_non_binary_inputs():
    truth_table_data = {
        "input_port": ["00", "01", "10", "11"],
        "output_port": ["00", "10", "11", "01"],
    }
    truth_table = TruthTable(
        input_ports=["input_port"], output_ports=["output_port"], **truth_table_data
    )

    am_module = construct_amaranth_module_from_truth_table(
        truth_table, logic_implementation_type="combinatorial"
    )

    assert isinstance(am_module, am.Elaboratable)

    # Test the module with non-binary inputs to see how it handles them
    # In this test, we only check that the module gets created without errors


def test_sequential_truth_table():
    truth_table_data = {
        "input_port": ["00", "01", "10", "11"],
        "output_port": ["00", "10", "11", "01"],
    }
    truth_table = TruthTable(
        input_ports=["input_port"], output_ports=["output_port"], **truth_table_data
    )

    # For now, we simulate only combinatorial. Implementing sequential and memory types would be more complex.
    am_module = construct_amaranth_module_from_truth_table(
        truth_table, logic_implementation_type="sequential"
    )

    assert isinstance(am_module, am.Elaboratable)

    # For sequential, a detailed simulation handling clock and state would be required.
    # Here, we check that the module is created correctly.
