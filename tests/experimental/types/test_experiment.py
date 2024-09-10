import pytest
import pandas as pd
from piel.types.experimental import (
    ExperimentInstance,
    Experiment,
    ExperimentCollection,
    PropagationDelayMeasurementConfiguration,
)
from piel.types import (
    PhysicalConnection,
    PhysicalComponent,
)


# Fixtures for the measurement used in ExperimentInstance and Experiment
@pytest.fixture
def mock_component_types():
    return PhysicalComponent()


@pytest.fixture
def mock_connection_types():
    return PhysicalConnection()


@pytest.fixture
def mock_measurement_configuration_types():
    return PropagationDelayMeasurementConfiguration()


@pytest.fixture
def mock_path_types():
    return "/mock/path"


# Test ExperimentInstance
def test_experiment_instance_initialization(
    mock_component_types, mock_connection_types, mock_measurement_configuration_types
):
    instance = ExperimentInstance(
        components=[mock_component_types],
        connections=[mock_connection_types],
        goal="Test Goal",
        parameters={"param1": "value1"},
        index=1,
        date_configured="2024-08-15",
        date_measured="2024-08-16",
        measurement_configuration_list=[mock_measurement_configuration_types],
    )

    assert instance.components == [mock_component_types]
    assert instance.connections == [mock_connection_types]
    assert instance.goal == "Test Goal"
    assert instance.parameters == {"param1": "value1"}
    assert instance.index == 1
    assert instance.date_configured == "2024-08-15"
    assert instance.date_measured == "2024-08-16"
    assert instance.measurement_configuration_list == [
        mock_measurement_configuration_types
    ]


def test_experiment_instance_default_initialization():
    instance = ExperimentInstance()

    assert instance.components == []
    assert instance.connections == []
    assert instance.goal == ""
    assert instance.parameters == {}
    assert instance.index == 0
    assert instance.date_configured == ""
    assert instance.date_measured == ""
    assert instance.measurement_configuration_list == []


# Test Experiment
def test_experiment_initialization(mock_path_types):
    instance_1 = ExperimentInstance(goal="Instance 1")
    instance_2 = ExperimentInstance(goal="Instance 2")
    experiment = Experiment(
        name="Test Experiment",
        goal="Overall Goal",
        experiment_instances=[instance_1, instance_2],
        parameters_list=[{"param1": "value1"}, {"param2": "value2"}],
        parent_directory=mock_path_types,
    )

    assert experiment.name == "Test Experiment"
    assert experiment.goal == "Overall Goal"
    assert experiment.experiment_instances == [instance_1, instance_2]
    assert experiment.parameters_list == [{"param1": "value1"}, {"param2": "value2"}]
    assert experiment.parent_directory == mock_path_types
    assert isinstance(experiment.parameters, pd.DataFrame)


def test_experiment_default_initialization():
    experiment = Experiment()

    assert experiment.name == ""
    assert experiment.goal == ""
    assert experiment.experiment_instances == []
    assert experiment.parameters_list == [{}]
    assert experiment.parent_directory == ""
    assert isinstance(experiment.parameters, pd.DataFrame)


# Test ExperimentCollection
def test_experiment_collection_initialization():
    experiment_1 = Experiment(name="Experiment 1")
    experiment_2 = Experiment(name="Experiment 2")
    collection = ExperimentCollection(experiment_list=[experiment_1, experiment_2])

    assert collection.experiment_list == [experiment_1, experiment_2]


def test_experiment_collection_default_initialization():
    collection = ExperimentCollection()

    assert collection.experiment_list == []


# Add more tests as needed to cover additional methods and edge cases
