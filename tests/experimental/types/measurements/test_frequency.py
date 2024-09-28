import pytest
import pathlib
import os
import types
from piel.types.experimental import (
    VNASParameterMeasurementConfiguration,
    VNAPowerSweepMeasurementConfiguration,
    VNASParameterMeasurement,
    VNAPowerSweepMeasurement,
    VNASParameterMeasurementCollection,
    VNAPowerSweepMeasurementCollection,
)


# Test VNASParameterMeasurementConfiguration
def test_vna_sparameter_measurement_configuration_initialization():
    config = VNASParameterMeasurementConfiguration(
        frequency_range_Hz=(1e6, 1e9), sweep_points=1001, test_port_power_dBm=-10.0
    )
    assert (
        config.measurement_configuration_type == "VNASParameterMeasurementConfiguration"
    )
    assert config.frequency_range_Hz == (1e6, 1e9)
    assert config.sweep_points == 1001
    assert config.test_port_power_dBm == -10.0


def test_vna_sparameter_measurement_configuration_default_initialization():
    config = VNASParameterMeasurementConfiguration()
    assert (
        config.measurement_configuration_type == "VNASParameterMeasurementConfiguration"
    )
    assert config.frequency_range_Hz == []
    assert config.sweep_points == 0
    assert config.test_port_power_dBm == 0.0


# Test VNAPowerSweepMeasurementConfiguration
def test_vna_power_sweep_measurement_configuration_initialization():
    config = VNAPowerSweepMeasurementConfiguration(
        base_frequency_Hz=2.4e9, power_range_dBm=(-30.0, 0.0)
    )
    assert (
        config.measurement_configuration_type == "VNAPowerSweepMeasurementConfiguration"
    )
    assert config.base_frequency_Hz == 2.4e9
    assert config.power_range_dBm == (-30.0, 0.0)


def test_vna_power_sweep_measurement_configuration_default_initialization():
    config = VNAPowerSweepMeasurementConfiguration()
    assert (
        config.measurement_configuration_type == "VNAPowerSweepMeasurementConfiguration"
    )
    assert config.base_frequency_Hz == 0.0
    assert config.power_range_dBm == []


# Test VNASParameterMeasurement with different PathTypes
@pytest.mark.parametrize(
    "path_type",
    [
        "/path/to/spectrum_file",  # str
        pathlib.Path("/path/to/spectrum_file"),  # pathlib.Path
        os.path,  # os.PathLike (os module itself in this case)
        types.ModuleType("mock_module"),  # measurement.ModuleType
    ],
)
def test_vna_sparameter_measurement_initialization(path_type):
    measurement = VNASParameterMeasurement(spectrum_file=path_type)
    assert measurement.type == "VNASParameterMeasurement"
    assert measurement.spectrum_file == path_type


def test_vna_sparameter_measurement_default_initialization():
    measurement = VNASParameterMeasurement()
    assert measurement.type == "VNASParameterMeasurement"
    assert measurement.spectrum_file == ""


# Test VNAPowerSweepMeasurement with different PathTypes
@pytest.mark.parametrize(
    "path_type",
    [
        "/path/to/spectrum_file",  # str
        pathlib.Path("/path/to/spectrum_file"),  # pathlib.Path
        os.path,  # os.PathLike (os module itself in this case)
        types.ModuleType("mock_module"),  # measurement.ModuleType
    ],
)
def test_vna_power_sweep_measurement_initialization(path_type):
    measurement = VNAPowerSweepMeasurement(spectrum_file=path_type)
    assert measurement.type == "VNAPowerSweepMeasurement"
    assert measurement.spectrum_file == path_type


def test_vna_power_sweep_measurement_default_initialization():
    measurement = VNAPowerSweepMeasurement()
    assert measurement.type == "VNAPowerSweepMeasurement"
    assert measurement.spectrum_file == ""


# Test VNASParameterMeasurementCollection
def test_vna_sparameter_measurement_collection_initialization():
    measurement_1 = VNASParameterMeasurement()
    measurement_2 = VNASParameterMeasurement()
    collection = VNASParameterMeasurementCollection(
        collection=[measurement_1, measurement_2]
    )
    assert collection.type == "VNASParameterMeasurementCollection"
    assert collection.collection == [measurement_1, measurement_2]


def test_vna_sparameter_measurement_collection_default_initialization():
    collection = VNASParameterMeasurementCollection()
    assert collection.type == "VNASParameterMeasurementCollection"
    assert collection.collection == []


# Test VNAPowerSweepMeasurementCollection
def test_vna_power_sweep_measurement_collection_initialization():
    subcollection_1 = VNAPowerSweepMeasurement()
    subcollection_2 = VNAPowerSweepMeasurement()
    collection = VNAPowerSweepMeasurementCollection(
        collection=[subcollection_1, subcollection_2]
    )
    assert collection.type == "VNAPowerSweepMeasurementCollection"
    assert collection.collection == [subcollection_1, subcollection_2]


def test_vna_power_sweep_measurement_collection_default_initialization():
    collection = VNAPowerSweepMeasurementCollection()
    assert collection.type == "VNAPowerSweepMeasurementCollection"
    assert collection.collection == []


# Add more tests as needed for additional methods, edge cases, and behaviors.
