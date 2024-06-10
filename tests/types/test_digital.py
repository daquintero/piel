import pandas as pd
from piel.types import (
    TruthTable,
)  # Adjust the import based on your actual module structure


def test_truth_table_initialization():
    input_ports = ["A", "B"]
    output_ports = ["Q"]
    truth_table_data = {
        "input_ports": input_ports,
        "output_ports": output_ports,
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        "Q": [0, 1, 1, 0],
    }
    tt = TruthTable(**truth_table_data)
    assert tt.input_ports == input_ports
    assert tt.output_ports == output_ports


def test_truth_table_ports_list():
    input_ports = ["A", "B"]
    output_ports = ["Q"]
    tt = TruthTable(input_ports=input_ports, output_ports=output_ports)
    assert tt.ports_list == input_ports + output_ports


def test_truth_table_dataframe_property():
    truth_table_data = {
        "input_ports": ["A", "B"],
        "output_ports": ["Q"],
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        "Q": [0, 1, 1, 0],
    }
    tt = TruthTable(**truth_table_data)
    df = tt.dataframe

    assert isinstance(df, pd.DataFrame)
    assert "A" in df.columns
    assert "B" in df.columns
    assert "Q" in df.columns
    assert df["A"].tolist() == truth_table_data["A"]
    assert df["B"].tolist() == truth_table_data["B"]
    assert df["Q"].tolist() == truth_table_data["Q"]


def test_truth_table_implementation_dictionary():
    input_ports = ["A", "B"]
    output_ports = ["Q"]
    truth_table_data = {
        "input_ports": input_ports,
        "output_ports": output_ports,
        "A": [0, 1],
        "B": [0, 1],
        "Q": [0, 1],
        "extra": "not included in implementation dictionary",
    }
    tt = TruthTable(**truth_table_data)
    impl_dict = tt.implementation_dictionary

    assert isinstance(impl_dict, dict)
    assert "A" in impl_dict
    assert "B" in impl_dict
    assert "Q" in impl_dict
    assert "extra" not in impl_dict
    assert impl_dict["A"] == truth_table_data["A"]
    assert impl_dict["B"] == truth_table_data["B"]
    assert impl_dict["Q"] == truth_table_data["Q"]
