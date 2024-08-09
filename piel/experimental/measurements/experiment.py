from typing import get_origin
from ..types import (
    Experiment,
    ExperimentInstance,
    MeasurementTypes,
    MeasurementCollectionTypes,
)
from .map import (
    configuration_to_measurement_map,
    measurement_composition_method_mapping,
    measurement_to_collection_map,
)
from ...types import PathTypes
from ...file_system import return_path


def compose_measurement_from_experiment_instance(
    experiment_instance: ExperimentInstance,
    instance_directory: PathTypes,
    configuration_measurement_map: dict = configuration_to_measurement_map,
    measurement_composition_method_mapping: dict = measurement_composition_method_mapping,
    **kwargs,
) -> MeasurementTypes:
    """
    This function is meant to be run after the measurements have been collected and the data exists within the measurement directories.
    Each experiment instance should correspond to a list of measurement configurations, ie the specific set of measurements
     that are required at each directory generated for the experimental instance. Hence, for this function to work properly,
     it is required to have a mapping between experiment configuration types and measurement classes accordingly.
     The mapping will be between a given ``MeasurementConfiguration`` type and a ``Measurement`` class which has the references
     of the data containers accordingly.
    """
    # This corresponds to the instance directory
    instance_directory = return_path(instance_directory)
    # TODO verify that print(experiment_instance.measurement_configuration_list) exists

    for (
        measurement_configuration_i
    ) in experiment_instance.measurement_configuration_list:
        # Now we need to go through the instance directory and map these files to a specific measurement instance
        measurement_composition_method = measurement_composition_method_mapping[
            measurement_configuration_i.__class__.__name__
        ]
        measurement = measurement_composition_method(
            instance_directory, name=experiment_instance.name, **kwargs
        )

    return measurement


def compose_measurement_collection_from_experiment(
    experiment: Experiment, experiment_directory: PathTypes, **kwargs
) -> MeasurementCollectionTypes:
    """
    This function takes a defined experiment and returns a measurement collection from them.
    Note that the complexity of this is verifying that the experiment is composed of the same type of measurements accordingly.
    TODO this should be validated in the Experiment composition accordingly.
    """
    experiment_directory = return_path(experiment_directory)
    measurement_collection_list: MeasurementCollectionTypes = list()

    instance_directory_index_i = 0
    for experiment_instance_i in experiment.experiment_instances:
        instance_directory = experiment_directory / str(instance_directory_index_i)

        measurement_i = compose_measurement_from_experiment_instance(
            experiment_instance=experiment_instance_i,
            instance_directory=instance_directory,
            **kwargs,
        )
        measurement_collection_list.append(measurement_i)
        # TODO implement measurement to measurement collection mapping
        instance_directory_index_i += 1

    # Use the last element to verify the collection map
    MeasurementCollectionType = measurement_to_collection_map[
        measurement_i.__class__.__name__
    ]
    assert isinstance(
        measurement_collection_list, get_origin(MeasurementCollectionType)
    )
    return measurement_collection_list
