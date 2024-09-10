from piel.types.experimental import (
    Experiment,
    ExperimentInstance,
    ExperimentCollection,
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
from ...models import load_from_json, load_from_dict


def compose_measurement_from_experiment_instance(
    experiment_instance: ExperimentInstance,
    instance_directory: PathTypes,
    configuration_measurement_map: dict = configuration_to_measurement_map,
    measurement_composition_method_mapping: dict = measurement_composition_method_mapping,
    composition_methods_kwargs: dict = None,
    **kwargs,
) -> MeasurementTypes:
    """
    This function is meant to be run after the measurements have been collected and the data exists within the measurement directories.
    Each experiment instance should correspond to a list of measurement configurations, ie the specific set of measurements
     that are required at each directory generated for the measurement instance. Hence, for this function to work properly,
     it is required to have a mapping between experiment configuration measurement and measurement classes accordingly.
     The mapping will be between a given ``MeasurementConfiguration`` type and a ``Measurement`` class which has the references
     of the data containers accordingly.
    """
    if composition_methods_kwargs is None:
        composition_methods_kwargs = {}

    # This corresponds to the instance directory
    instance_directory = return_path(instance_directory)
    # TODO verify that print(experiment_instance.measurement_configuration_list) exists

    experiment_instance_dict = experiment_instance.model_dump()

    measurement_configuration_list = experiment_instance_dict[
        "measurement_configuration_list"
    ]

    if len(measurement_configuration_list) == 0:
        # Include information about which experiment or directory this is.
        raise ValueError(
            "The experiment instance does not contain any measurements. Please verify the experiment configuration."
            + f"Experiment instance: {experiment_instance.name} "
            + f"Directory: {instance_directory}"
        )
    else:
        for measurement_configuration_i in measurement_configuration_list:
            # Now we need to go through the instance directory and map these files to a specific measurement instance
            measurement_composition_method = measurement_composition_method_mapping[
                measurement_configuration_i["measurement_configuration_type"]
            ]
            measurement = measurement_composition_method(
                instance_directory,
                name=experiment_instance.name,
                **composition_methods_kwargs,
            )
            # TODO fix me

            return measurement


def compose_measurement_collection_from_experiment(
    experiment: Experiment, experiment_directory: PathTypes = None, **kwargs
) -> MeasurementCollectionTypes:
    """
    This function takes a defined experiment and returns a measurement collection from them.
    Note that the complexity of this is verifying that the experiment is composed of the same type of measurements accordingly.
    TODO this should be validated in the Experiment composition accordingly.
    """
    experiment_directory = return_path(experiment_directory)
    measurement_collection_list: MeasurementCollectionTypes = list()
    experiment_dict = experiment.model_dump()

    instance_directory_index_i = 0
    for experiment_instance_i in experiment_dict["experiment_instances"]:
        instance_directory = experiment_directory / str(instance_directory_index_i)

        experiment_instance_i = load_from_dict(
            experiment_instance_i, ExperimentInstance
        )

        measurement_i = compose_measurement_from_experiment_instance(
            experiment_instance=experiment_instance_i,
            instance_directory=instance_directory,
            **kwargs,
        )
        measurement_collection_list.append(measurement_i)
        # TODO implement measurement to measurement collection mapping
        instance_directory_index_i += 1

    if len(measurement_collection_list) == 0:
        raise ValueError(
            "The experiment does not contain any measurements. Please verify the experiment configuration."
            + f"Experiment instance: {experiment.name} "
            + f"Directory: {experiment_directory}"
        )
    else:
        # Use the first element to verify the collection map
        measurement_collection_type = measurement_to_collection_map[
            measurement_collection_list[0].type
        ]
        return measurement_collection_type(collection=measurement_collection_list)


def load_from_directory(
    parent_directory: PathTypes,
) -> ExperimentInstance | Experiment:
    """
    This function is inputted a directory, and the aim is to read the corresponding metadata to extract
    the model definition used to re-instantiate the class instance a Python object. This is useful for
    when we want to load a previously saved experiment instance.

    Parameters
    ----------
    parent_directory : PathTypes
        The directory where the metadata file is located.

    Returns
    -------
    ExperimentInstance | Experiment
        The corresponding experiment instance or experiment object
    """
    experiment_json = parent_directory / "experiment.json"
    instance_json = parent_directory / "instance.json"

    if experiment_json.exists():
        model_instance = load_from_json(experiment_json, Experiment)
    elif instance_json.exists():
        model_instance = load_from_json(instance_json, ExperimentInstance)
    else:
        raise FileNotFoundError(
            f"Could not find the corresponding metadata file in {parent_directory} to reconstruct the model class."
        )
    return model_instance


def load_experiment_collection() -> ExperimentCollection:
    """
    This function is provided a directory which contains subdirectories of experiments. It will load all the experiments
    and return an `ExperimentCollection` object.
    """
