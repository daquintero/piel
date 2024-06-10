import pytest
import amaranth as am
from amaranth.back import verilog
from piel.tools.amaranth import generate_verilog_from_amaranth_truth_table
from piel.types import TruthTable
import pathlib
import types
import os


# Helper function to create a dummy Amaranth module
class SimpleAmaranthModule(am.Elaboratable):
    def __init__(self):
        self.input1 = am.Signal(2)
        self.input2 = am.Signal(2)
        self.output1 = am.Signal()

    def elaborate(self, platform):
        m = am.Module()
        m.d.comb += self.output1.eq(self.input1[0] & self.input2[0])
        return m


# Tests for generate_verilog_from_amaranth_truth_table function
def test_generate_verilog(tmp_path):
    # Define a simple truth table
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "input2": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1", "input2"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Call the function to generate Verilog
    target_file_name = "output.v"
    target_directory = tmp_path
    generate_verilog_from_amaranth_truth_table(
        am_module, truth_table, target_file_name, target_directory
    )

    # Check that the file was created
    target_file_path = target_directory / target_file_name
    assert target_file_path.exists()

    # Check that the file contains Verilog code
    with target_file_path.open("r") as f:
        verilog_code = f.read()
        assert "module " in verilog_code  # Check for general module declaration
        assert "input1" in verilog_code
        assert "input2" in verilog_code
        assert "output1" in verilog_code


def test_generate_verilog_with_missing_port(tmp_path):
    # Define a truth table with a missing port
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1", "missing_input"],
        output_ports=["output1"],
        **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Call the function and expect it to raise an AttributeError
    target_file_name = "output.v"
    target_directory = tmp_path
    with pytest.raises(
        AttributeError, match="Port missing_input not found in the Amaranth module"
    ):
        generate_verilog_from_amaranth_truth_table(
            am_module, truth_table, target_file_name, target_directory
        )


# TODO fix this
# def test_generate_verilog_in_module_path(tmp_path, monkeypatch):
#     # Define a simple truth table
#     truth_table_data = {
#         "input1": ["00", "01", "10", "11"],
#         "input2": ["00", "01", "10", "11"],
#         "output1": ["0", "0", "0", "1"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input1", "input2"],
#         output_ports=["output1"],
#         **truth_table_data
#     )
#
#     # Create a simple Amaranth module
#     am_module = SimpleAmaranthModule()
#
#     # Mock the get_module_folder_type_location to return a specific path
#     module_path = tmp_path / "module_directory"
#     module_path.mkdir()
#     source_directory = module_path / "src"
#     source_directory.mkdir()
#
#     def mock_get_module_folder_type_location(module, folder_type):
#         return source_directory
#
#     # Adjust the monkeypatch target to match the actual import path
#     monkeypatch.setattr("your_actual_module_path.get_module_folder_type_location", mock_get_module_folder_type_location)
#
#     # Create a dummy module
#     dummy_module = types.ModuleType("dummy_module")
#     dummy_module.__file__ = str(module_path / "dummy_file.py")
#
#     # Call the function to generate Verilog
#     target_file_name = "output.v"
#     generate_verilog_from_amaranth_truth_table(
#         am_module, truth_table, target_file_name, dummy_module
#     )
#
#     # Check that the file was created in the mocked directory
#     target_file_path = source_directory / target_file_name
#     assert target_file_path.exists()


def test_generate_verilog_without_target_directory(tmp_path):
    # Define a simple truth table
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "input2": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1", "input2"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Create the target file path without directory
    target_file_name = "output.v"
    target_directory = tmp_path / "non_existent_directory"

    # Expect the function to create the directory
    generate_verilog_from_amaranth_truth_table(
        am_module, truth_table, target_file_name, target_directory
    )

    # Check that the directory and file were created
    assert target_directory.exists()
    assert (target_directory / target_file_name).exists()


def test_generate_verilog_custom_backend(tmp_path, monkeypatch):
    # Define a simple truth table
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "input2": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1", "input2"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Define a mock backend
    class MockBackend:
        @staticmethod
        def convert(*args, **kwargs):
            return "// Mock Verilog code\nmodule MockModule();\nendmodule\n"

    target_file_name = "output.v"
    target_directory = tmp_path

    # Call the function with the custom backend
    generate_verilog_from_amaranth_truth_table(
        am_module, truth_table, target_file_name, target_directory, backend=MockBackend
    )

    # Check that the file was created and contains the mock Verilog code
    target_file_path = target_directory / target_file_name
    assert target_file_path.exists()

    with target_file_path.open("r") as f:
        verilog_code = f.read()
        assert "// Mock Verilog code" in verilog_code
        assert "module MockModule" in verilog_code
