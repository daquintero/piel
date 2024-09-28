from typing import Union
import pandas as pd


def index_experiment(instance, index: Union[int, slice, pd.DataFrame]):
    """
    Indexes an Experiment instance based on an integer, slice, or pandas DataFrame.
    Returns a new Experiment instance with the indexed subset of relevant attributes,
    preserving all other attributes.

    Args:
        instance: The Experiment instance to index.
        index: An integer index, slice, or pandas DataFrame specifying the subset.

    Returns:
        A new Experiment instance with the indexed attributes.
    """
    # Create a copy of the instance's attributes
    attrs = vars(instance).copy()

    if isinstance(index, pd.DataFrame):
        # Assume the DataFrame's index corresponds to the indices of experiment_instances and parameters_list
        # Extract the indices from the DataFrame
        if not index.index.is_integer():
            raise ValueError(
                "DataFrame index must be integer-based to correspond with list indices."
            )
        indices = index.index.tolist()
    elif isinstance(index, (int, slice)):
        # Handle integer and slice indexing
        if isinstance(index, int):
            indices = [index]
        else:
            indices = list(range(*index.indices(len(instance.experiment_instances))))
    else:
        raise TypeError("Index must be an integer, slice, or pandas DataFrame.")

    # Validate indices
    max_index = len(instance.experiment_instances) - 1
    for idx in indices:
        if idx < 0 or idx > max_index:
            raise IndexError(
                f"Index {idx} out of range for experiment_instances with length {len(instance.experiment_instances)}."
            )

    # Index experiment_instances and parameters_list
    attrs["experiment_instances"] = [instance.experiment_instances[i] for i in indices]
    attrs["parameters_list"] = [instance.parameters_list[i] for i in indices]

    # Create and return a new Experiment instance with updated attributes
    return instance.__class__(**attrs)
