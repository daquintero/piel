import pytest
import pathlib
import os
import types
from piel.types.experimental import (
    MeasurementConfiguration,
    Measurement,
    MeasurementCollection,
)
from piel.types import PielBaseModel


# Test MeasurementConfiguration
def test_measurement_configuration_initialization():
    config = MeasurementConfiguration(
        name="Test Measurement Configuration",
        parent_directory="/path/to/parent_directory",
        measurement_type="Test Type",
    )
    assert isinstance(config, PielBaseModel)
    assert config.name == "Test Measurement Configuration"
    assert config.parent_directory == "/path/to/parent_directory"
    assert config.measurement_type == "Test Type"


def test_measurement_configuration_default_initialization():
    config = MeasurementConfiguration()
    assert config.name == ""
    assert config.parent_directory == ""
    assert config.measurement_type == ""


# Test Measurement with different PathTypes
@pytest.mark.parametrize(
    "path_type",
    [
        "/path/to/parent_directory",  # str
        pathlib.Path("/path/to/parent_directory"),  # pathlib.Path
        os.path,  # os.PathLike (os module itself in this case)
        types.ModuleType("mock_module"),  # measurement.ModuleType
    ],
)
def test_measurement_initialization(path_type):
    measurement = Measurement(
        name="Test Measurement", type="Test Type", parent_directory=path_type
    )
    assert isinstance(measurement, PielBaseModel)
    assert measurement.name == "Test Measurement"
    assert measurement.type == "Test Type"
    assert measurement.parent_directory == path_type


def test_measurement_default_initialization():
    measurement = Measurement()
    assert measurement.name == ""
    assert measurement.type == ""
    assert measurement.parent_directory == ""


# Test MeasurementCollection
def test_measurement_collection_initialization():
    measurement_1 = Measurement(name="Measurement 1")
    measurement_2 = Measurement(name="Measurement 2")
    collection = MeasurementCollection(
        type="Test Collection", collection=[measurement_1, measurement_2]
    )
    assert isinstance(collection, PielBaseModel)
    assert collection.type == "Test Collection"
    assert collection.collection == [measurement_1, measurement_2]


def test_measurement_collection_default_initialization():
    collection = MeasurementCollection()
    assert collection.type == ""
    assert collection.collection == []


# Add more tests as needed for additional methods, edge cases, and behaviors.
