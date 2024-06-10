import numpy as np
from piel.parametric import (
    single_parameter_sweep,
    multi_parameter_sweep,
)  # Adjust the import based on your actual module structure


# Tests for single_parameter_sweep function
def test_single_parameter_sweep_basic():
    base_config = {"param1": 10, "param2": 20}
    param_name = "param1"
    sweep_values = [1, 2, 3]

    expected_output = [
        {"param1": 1, "param2": 20},
        {"param1": 2, "param2": 20},
        {"param1": 3, "param2": 20},
    ]

    result = single_parameter_sweep(base_config, param_name, sweep_values)
    assert result == expected_output


def test_single_parameter_sweep_empty_values():
    base_config = {"param1": 10, "param2": 20}
    param_name = "param1"
    sweep_values = []

    expected_output = []

    result = single_parameter_sweep(base_config, param_name, sweep_values)
    assert result == expected_output


def test_single_parameter_sweep_non_existing_parameter():
    base_config = {"param1": 10, "param2": 20}
    param_name = "param3"  # param3 does not exist in base_config
    sweep_values = [1, 2, 3]

    expected_output = [
        {"param1": 10, "param2": 20, "param3": 1},
        {"param1": 10, "param2": 20, "param3": 2},
        {"param1": 10, "param2": 20, "param3": 3},
    ]

    result = single_parameter_sweep(base_config, param_name, sweep_values)
    assert result == expected_output


# Tests for multi_parameter_sweep function
def test_multi_parameter_sweep_basic():
    base_config = {"param1": 10, "param2": 20}
    sweep_dict = {"param1": np.array([1, 2]), "param2": np.array([3, 4])}

    expected_output = [
        {"param1": 1, "param2": 3},
        {"param1": 1, "param2": 4},
        {"param1": 2, "param2": 3},
        {"param1": 2, "param2": 4},
    ]

    result = multi_parameter_sweep(base_config, sweep_dict)
    assert result == expected_output


def test_multi_parameter_sweep_single_parameter():
    base_config = {"param1": 10, "param2": 20}
    sweep_dict = {"param1": np.array([1, 2, 3])}

    expected_output = [
        {"param1": 1, "param2": 20},
        {"param1": 2, "param2": 20},
        {"param1": 3, "param2": 20},
    ]

    result = multi_parameter_sweep(base_config, sweep_dict)
    assert result == expected_output


def test_multi_parameter_sweep_empty_values():
    base_config = {"param1": 10, "param2": 20}
    sweep_dict = {"param1": np.array([]), "param2": np.array([])}

    expected_output = []

    result = multi_parameter_sweep(base_config, sweep_dict)
    assert result == expected_output


# TODO fix this
# def test_multi_parameter_sweep_empty_dict():
#     base_config = {"param1": 10, "param2": 20}
#     sweep_dict = {}
#
#     expected_output = []
#
#     result = multi_parameter_sweep(base_config, sweep_dict)
#     assert result == expected_output


def test_multi_parameter_sweep_additional_parameter():
    base_config = {"param1": 10, "param2": 20}
    sweep_dict = {"param1": np.array([1, 2]), "param3": np.array([30, 40])}

    expected_output = [
        {"param1": 1, "param2": 20, "param3": 30},
        {"param1": 1, "param2": 20, "param3": 40},
        {"param1": 2, "param2": 20, "param3": 30},
        {"param1": 2, "param2": 20, "param3": 40},
    ]

    result = multi_parameter_sweep(base_config, sweep_dict)
    assert result == expected_output
