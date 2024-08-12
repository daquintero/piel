from ...utils import get_unique_dataframe_subsets
from ..types import Experiment, ExperimentData, ExperimentDataCollection
from ..measurements.map import measurement_data_to_measurement_collection_data_map


def create_experiment_data_collection_from_unique_parameters(
    experiment_data: ExperimentData,
) -> ExperimentDataCollection:
    """
    Each individual raw ``ExperimentData`` can contain multiple operating points or unique parameters which are being
    tested. It can be handy to create subsets of ``ExperimentData`` -> multiple ``ExperimentData``s
    that correspond to relevant operating points stored with the relevant parameters both internally and in an
    ``ExperimentDataCollection``. As such, it is easier to understand the collection of data
    measurements based on this and perform plotting accordingly in a more relevant implementation. Likewise,
    the corresponding operating point metadata is encoded in the generated ``ExperimentData`` sets.

    First, we will need to extract the operating points from the ``ExperimentData.experiment.parameters``.
    This can be done by identifying the unique elements from the pandas DataFrame.
    Then, we will need to create a new ``ExperimentData`` for each of the operating points.
    Finally, we will need to create a new ``ExperimentDataCollection`` with the new ``ExperimentData``.
    """
    # First we extract the relevant parameter subsets
    unique_parameter_subset_dictionary = get_unique_dataframe_subsets(
        experiment_data.experiment.parameters
    )

    # Then we create a new ExperimentData for each of the operating points, we also need to create a new Experiment within the ExperimentData to match the indices
    experiment_data_collection = list()
    for (
        identifier_i,
        unique_parameter_subset_i,
    ) in unique_parameter_subset_dictionary.items():
        experiment_i = experiment_data.experiment

        # We also need to create a new Experiment within the ExperimentData to match the indices
        experiment_instances_subset_i = [
            experiment_i.experiment_instances[i]
            for i in unique_parameter_subset_i.index
        ]

        # Create sub experiment accordingly
        experiment_i = Experiment(
            name=f"{experiment_i.name}_{identifier_i}",
            goal=f"{experiment_i.goal}_{identifier_i}",
            experiment_instances=experiment_instances_subset_i,
            parameters_list=unique_parameter_subset_i.to_dict(orient="records"),
        )

        # Create subset of experiment data
        experiment_data_instances_subset_i = [
            experiment_data.data.collection[i] for i in unique_parameter_subset_i.index
        ]

        # Create the measurement collection with the correct type.
        collection_type = measurement_data_to_measurement_collection_data_map[
            experiment_data_instances_subset_i[-1].__class__.__name__
        ]

        # Create the corresponding ExperimentData
        data_collection = collection_type(
            name=f"{experiment_i.name}_{identifier_i}",
            collection=experiment_data_instances_subset_i,
        )

        experiment_data_i = ExperimentData(
            name=experiment_i.name,
            experiment=experiment_i,
            data=data_collection,
        )

        experiment_data_collection.append(experiment_data_i)

    # Create the corresponding ExperimentDataCollection
    experiment_data_collection = ExperimentDataCollection(
        name=experiment_data.name, collection=experiment_data_collection
    )

    return experiment_data_collection
