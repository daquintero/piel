from piel.types import ExperimentData, MeasurementDataCollection
import pandas as pd
from copy import deepcopy


def experiment_data_from_parameter_subset(
    experiment_data: ExperimentData, subset: pd.DataFrame
) -> ExperimentData:
    """
    Filters the given ExperimentData based on a subset of parameters.

    Args:
        experiment_data (ExperimentData): The original experiment data.
        subset (pd.DataFrame): The subset of parameters to filter by.

    Returns:
        ExperimentData: A new ExperimentData instance containing only the filtered data.
    """

    # Ensure the subset indices align with parameters_list
    subset_indices = subset.index.tolist()

    # Debug: Print subset indices
    print(f"Filtering based on parameter subset indices: {subset_indices}")

    # Deepcopy to avoid mutating the original data
    new_experiment = deepcopy(experiment_data.experiment)
    new_data = deepcopy(experiment_data.data)

    # Filter the experiment_instances
    new_experiment.experiment_instances = [
        new_experiment.experiment_instances[i] for i in subset_indices
    ]

    # Filter the parameters_list
    new_experiment.parameters_list = [
        new_experiment.parameters_list[i] for i in subset_indices
    ]

    # Filter the MeasurementDataCollection
    if isinstance(new_data.collection, list):
        new_data.collection = [new_data.collection[i] for i in subset_indices]
    elif isinstance(new_data.collection, MeasurementDataCollection):
        # Assuming MeasurementDataCollection has a 'collection' attribute that's a list
        new_data.collection.collection = [
            new_data.collection.collection[i] for i in subset_indices
        ]
    else:
        raise TypeError("Unsupported MeasurementDataCollection type.")

    # Create and return the new ExperimentData
    filtered_experiment_data = ExperimentData(experiment=new_experiment, data=new_data)

    return filtered_experiment_data
