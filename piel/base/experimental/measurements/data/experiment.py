from typing import Union
import pandas as pd


def index_experiment_data(instance, index: Union[int, slice, pd.DataFrame]):
    """
    Indexes the data attribute of an ExperimentData instance based on an integer, slice, or pandas DataFrame.
    Returns a new ExperimentData instance with the indexed data, preserving all other attributes.

    If a pandas DataFrame is provided, it extracts the corresponding indices and uses them to index the data.

    Args:
        instance: The ExperimentData instance to index.
        index: An integer index, slice, or pandas DataFrame specifying the subset.

    Returns:
        A new ExperimentData instance with the indexed data.
    """
    # Create a copy of the instance's attributes
    attrs = vars(instance).copy()

    if isinstance(index, pd.DataFrame):
        # Assume the DataFrame's index corresponds to the indices of the data's collection
        if not index.index.is_integer():
            raise ValueError(
                "DataFrame index must be integer-based to correspond with data.collection indices."
            )
        indices = index.index.tolist()
    elif isinstance(index, (int, slice)):
        # Handle integer and slice indexing
        if isinstance(index, int):
            indices = [index]
        else:
            indices = list(range(*index.indices(len(instance.data.collection))))
    else:
        raise TypeError("Index must be an integer, slice, or pandas DataFrame.")

    # Validate indices
    max_index = len(instance.data.collection) - 1
    for idx in indices:
        if idx < 0 or idx > max_index:
            raise IndexError(
                f"Index {idx} out of range for data.collection with length {len(instance.data.collection)}."
            )

    # Index the data's collection
    new_data = instance.data[index]
    new_experiment = instance.experiment[index]

    # Update the 'data' attribute in the copied attributes
    attrs["data"] = new_data
    attrs["experiment"] = new_experiment

    # Create and return a new ExperimentData instance with updated attributes
    return instance.__class__(**attrs)
