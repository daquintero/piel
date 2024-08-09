from ...types import PathTypes
from ..types import (
    Experiment,
    ExperimentData,
)
from .experiment import compose_measurement_collection_from_experiment
from .data.extract import extract_data_from_measurement_collection


def extract_data_from_experiment(
    experiment: Experiment, experiment_directory: PathTypes, **kwargs
) -> ExperimentData:
    """
    This function must be run after data has already been written within the ``Experiment`` directories
    and the metadata has been created accordingly. This will extract all the corresponding measurements collection,
    and also extract the corresponding data from each setup accordingly. It will create a `ExperimentData` that collects
    both the metadata and measurement data.
    """
    measurement_collection = compose_measurement_collection_from_experiment(
        experiment=experiment,
        experiment_directory=experiment_directory,
    )

    experiment_data = extract_data_from_measurement_collection(
        measurement_collection=measurement_collection
    )

    return ExperimentData(experiment=experiment, data=experiment_data)
