from ....types import PathTypes
from ...types import (
    Experiment,
    ExperimentData,
)
from ....file_system import return_path
from ....models import load_from_json
from ..experiment import compose_measurement_collection_from_experiment
from ...types import MeasurementCollectionTypes, MeasurementDataCollectionTypes
from ..map import (
    measurement_to_data_map,
    measurement_to_data_method_map,
    measurement_data_to_measurement_collection_data_map,
)


def extract_data_from_measurement_collection(
    measurement_collection: MeasurementCollectionTypes,
    measurement_to_data_map: dict = measurement_to_data_map,
    measurement_to_data_method_map: dict = measurement_to_data_method_map,
    skip_missing: bool = False,
    **kwargs,
) -> MeasurementDataCollectionTypes:
    """
    The goal of this function is to compose the data from a collection of measurement references.
    Based on each type of measurement, it will apply an extraction function based on the data mapping accordingly.
    It will return a collection of data types which is inherent to the type of the measurement collection provided.
    """
    measurement_data_collection: MeasurementDataCollectionTypes = list()

    for measurement_i in measurement_collection.collection:
        # Identify correct data mapping
        measurement_data_type = measurement_to_data_map[measurement_i.type]
        extract_data_method = measurement_to_data_method_map[measurement_i.type]
        try:
            measurement_data_i = extract_data_method(measurement_i)
        except Exception as e:
            missing_data_error = f"Missing data for measurement: {measurement_i} in collection: {measurement_collection}"
            if skip_missing:
                print(missing_data_error)
                measurement_data_i = measurement_data_type()
            else:
                raise e
        assert isinstance(measurement_data_i, measurement_data_type)
        measurement_data_collection.append(measurement_data_i)

    # Now we need to extract the corresponding MeasurementCollection type from measurement_data_collection
    # Use the last element
    measurement_data_collection_type = (
        measurement_data_to_measurement_collection_data_map[measurement_data_i.type]
    )

    # Create the validated instance
    measurement_data_collection = measurement_data_collection_type(
        collection=measurement_data_collection
    )

    return measurement_data_collection


def extract_data_from_experiment(
    experiment: Experiment,
    experiment_directory: PathTypes,
    composition_kwargs: dict = None,
    extraction_kwargs: dict = None,
    **kwargs,
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
    if composition_kwargs is None:
        composition_kwargs = dict()

    if extraction_kwargs is None:
        extraction_kwargs = dict()

    measurement_collection = compose_measurement_collection_from_experiment(
        experiment=experiment,
        experiment_directory=experiment_directory,
        **composition_kwargs,
    )

    measurement_data_collection = extract_data_from_measurement_collection(
        measurement_collection=measurement_collection, **extraction_kwargs
    )

    return ExperimentData(
        experiment=experiment, data=measurement_data_collection, **kwargs
    )


def load_experiment_data_from_directory(
    experiment_directory: PathTypes, **kwargs
) -> ExperimentData:
    """
    This function will load an `Experiment` from the metadata stored in the `experiment.json` directory.
    """
    experiment_directory = return_path(experiment_directory)
    experiment_metadata_json = experiment_directory / "experiment.json"
    assert experiment_metadata_json.exists()
    experiment = load_from_json(experiment_metadata_json, Experiment)
    experiment_data = extract_data_from_experiment(
        experiment, experiment_directory=experiment_directory, **kwargs
    )
    return experiment_data
