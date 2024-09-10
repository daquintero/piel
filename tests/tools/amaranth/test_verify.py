import pytest
import amaranth as am
from piel.tools.amaranth import verify_amaranth_truth_table
from piel.types import TruthTable  # Adjust the import based on your actual module path


# Helper function to create a dummy Amaranth module
class SimpleAmaranthModule(am.Elaboratable):
    def __init__(self):
        self.input1 = am.Signal(2)
        self.output1 = am.Signal()

    def elaborate(self, platform):
        m = am.Module()
        m.d.comb += self.output1.eq(self.input1[0] & self.input1[1])
        return m


# Tests for verify_amaranth_truth_table function
def test_verify_combinatorial_logic(tmp_path):
    # Define a simple truth table
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Call the function to verify the truth table
    vcd_file_name = "output.vcd"
    target_directory = tmp_path
    verify_amaranth_truth_table(
        am_module,
        truth_table,
        vcd_file_name,
        target_directory,
        implementation_type="combinatorial",
    )

    # Check that the VCD file was created
    vcd_file_path = target_directory / vcd_file_name
    assert vcd_file_path.exists()


# TODO fix this
# def test_verify_sequential_logic(tmp_path):
#     # Define a simple truth table
#     truth_table_data = {
#         "input1": ["00", "01", "10", "11"],
#         "output1": ["0", "0", "0", "1"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input1"],
#         output_ports=["output1"],
#         **truth_table_data
#     )
#
#     # Create a simple Amaranth module
#     am_module = SimpleAmaranthModule()
#
#     # Call the function to verify the truth table with sequential type
#     vcd_file_name = "output_sequential.vcd"
#     target_directory = tmp_path
#     verify_amaranth_truth_table(
#         am_module, truth_table, vcd_file_name, target_directory, implementation_type="sequential"
#     )
#
#     # Check that the VCD file was created
#     vcd_file_path = target_directory / vcd_file_name
#     assert vcd_file_path.exists()
#
# def test_verify_with_missing_port(tmp_path):
#     # Define a truth table with a missing port
#     truth_table_data = {
#         "input1": ["00", "01", "10", "11"],
#         "output1": ["0", "0", "0", "1"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input1", "missing_input"],
#         output_ports=["output1"],
#         **truth_table_data
#     )
#
#     # Create a simple Amaranth module
#     am_module = SimpleAmaranthModule()
#
#     # Call the function and expect it to raise an AttributeError
#     vcd_file_name = "output.vcd"
#     target_directory = tmp_path
#     with pytest.raises(AttributeError, match="Port missing_input not found in the Amaranth module"):
#         verify_amaranth_truth_table(
#             am_module, truth_table, vcd_file_name, target_directory, implementation_type="combinatorial"
#         )


def test_verify_non_matching_output(tmp_path):
    # Define a truth table with non-matching output
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "output1": ["1", "0", "0", "0"],  # This will cause a mismatch
    }
    truth_table = TruthTable(
        input_ports=["input1"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Call the function and expect it to fail during simulation
    vcd_file_name = "output.vcd"
    target_directory = tmp_path

    with pytest.raises(AssertionError):
        verify_amaranth_truth_table(
            am_module,
            truth_table,
            vcd_file_name,
            target_directory,
            implementation_type="combinatorial",
        )


# TODO fix this
# def test_verify_vcd_generation_in_module_path(tmp_path, monkeypatch):
#     # Define a simple truth table
#     truth_table_data = {
#         "input1": ["00", "01", "10", "11"],
#         "output1": ["0", "0", "0", "1"]
#     }
#     truth_table = TruthTable(
#         input_ports=["input1"],
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
#     tb_directory = module_path / "tb"
#     tb_directory.mkdir()
#
#     def mock_get_module_folder_type_location(module, folder_type):
#         return tb_directory
#
#     # Adjust the monkeypatch target to match the actual import path
#     monkeypatch.setattr("your_actual_module_path.get_module_folder_type_location", mock_get_module_folder_type_location)
#
#     # Create a dummy module
#     dummy_module = measurement.ModuleType("dummy_module")
#     dummy_module.__file__ = str(module_path / "dummy_file.py")
#
#     # Call the function to verify the truth table
#     vcd_file_name = "output.vcd"
#     verify_amaranth_truth_table(
#         am_module, truth_table, vcd_file_name, dummy_module, implementation_type="combinatorial"
#     )
#
#     # Check that the VCD file was created in the mocked directory
#     vcd_file_path = tb_directory / vcd_file_name
#     assert vcd_file_path.exists()


def test_verify_memory_logic(tmp_path):
    # Define a simple truth table
    truth_table_data = {
        "input1": ["00", "01", "10", "11"],
        "output1": ["0", "0", "0", "1"],
    }
    truth_table = TruthTable(
        input_ports=["input1"], output_ports=["output1"], **truth_table_data
    )

    # Create a simple Amaranth module
    am_module = SimpleAmaranthModule()

    # Call the function to verify the truth table with memory type
    vcd_file_name = "output_memory.vcd"
    target_directory = tmp_path
    verify_amaranth_truth_table(
        am_module,
        truth_table,
        vcd_file_name,
        target_directory,
        implementation_type="memory",
    )

    # Check that the VCD file was created
    vcd_file_path = target_directory / vcd_file_name
    assert vcd_file_path.exists()
