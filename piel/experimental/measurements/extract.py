from ...types import PathTypes
from ..types import (
    Experiment,
    ExperimentData,
)
from ...file_system import return_path
from ...models import load_from_json
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

    Parameters
    ----------

    experiment : Experiment
        The experiment object that contains the metadata of the experiment.
    experiment_directory : PathTypes
        The directory where the experiment is located.
    **kwargs
        Extra keyword arguments passed to the class instantiation.

    Returns
    -------

    ExperimentData
        The data extracted from the experiment.
    """
    measurement_collection = compose_measurement_collection_from_experiment(
        experiment=experiment,
        experiment_directory=experiment_directory,
    )

    experiment_data = extract_data_from_measurement_collection(
        measurement_collection=measurement_collection
    )

    return ExperimentData(experiment=experiment, data=experiment_data, **kwargs)


def load_experiment_data_from_directory(
    experiment_directory: PathTypes,
) -> ExperimentData:
    """
    This function will load an `Experiment` from the metadata stored in the `experiment.json` directory.
    """
    experiment_directory = return_path(experiment_directory)
    experiment_metadata_json = experiment_directory / "experiment.json"
    assert experiment_metadata_json.exists()
    experiment = load_from_json(experiment_metadata_json, Experiment)
    experiment_data = extract_data_from_experiment(
        experiment, experiment_directory=experiment_directory
    )
    return experiment_data
