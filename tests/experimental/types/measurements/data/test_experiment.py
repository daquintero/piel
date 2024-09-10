from piel.types.experimental import (
    ExperimentData,
    ExperimentDataCollection,
    Experiment,
    PropagationDelayMeasurementDataCollection,
)
from piel.types import Instance


# Test ExperimentData
def test_experiment_data_initialization():
    experiment = Experiment(name="Test Experiment")
    measurement_data = (
        PropagationDelayMeasurementDataCollection()
    )  # Use appropriate mock or actual type

    experiment_data = ExperimentData(experiment=experiment, data=measurement_data)

    assert isinstance(experiment_data, Instance)
    assert experiment_data.experiment == experiment
    assert experiment_data.data == measurement_data


def test_experiment_data_default_initialization():
    experiment_data = ExperimentData()

    assert isinstance(experiment_data, Instance)
    assert experiment_data.experiment is None
    assert experiment_data.data is None


# Test ExperimentDataCollection
def test_experiment_data_collection_initialization():
    experiment_data_1 = ExperimentData()
    experiment_data_2 = ExperimentData()

    data_collection = ExperimentDataCollection(
        collection=[experiment_data_1, experiment_data_2]
    )

    assert isinstance(data_collection, Instance)
    assert data_collection.collection == [experiment_data_1, experiment_data_2]


def test_experiment_data_collection_default_initialization():
    data_collection = ExperimentDataCollection()

    assert isinstance(data_collection, Instance)
    assert data_collection.collection == []


# Add more tests as needed for validators, edge cases, and behaviors.
