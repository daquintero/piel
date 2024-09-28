from typing import Union, Any


def index_measurement_data_collection(instance: Any, index: Union[int, slice]) -> Any:
    """
    Allows indexing and slicing of the MeasurementDataCollection instance or its subclasses.
    Returns a new instance of the same class with the indexed subset of the collection,
    preserving all other attributes.

    Args:
        instance: The MeasurementDataCollection instance or subclass instance to index.
        index: An integer index or a slice object.

    Returns:
        A new instance of MeasurementDataCollection or its subclass with the indexed collection.
    """
    import pandas as pd

    if isinstance(index, pd.DataFrame):
        # Extract the indices from the DataFrame's index
        if not pd.api.types.is_integer_dtype(index.index):
            raise ValueError(
                "DataFrame index must be integer-based to correspond with collection indices."
            )
        indices = index.index.tolist()
    elif isinstance(index, int):
        indices = [index]
    elif isinstance(index, slice):
        indices = list(range(*index.indices(len(instance.collection))))
    else:
        raise TypeError("Index must be an integer, slice, or pandas DataFrame.")

    # Validate indices
    max_index = len(instance.collection) - 1
    for idx in indices:
        if idx < 0 or idx > max_index:
            raise IndexError(
                f"Index {idx} out of range for collection with length {len(instance.collection)}."
            )

    # Index the collection
    if isinstance(index, int):
        indexed_subset = [instance.collection[index]]
    else:
        indexed_subset = [instance.collection[i] for i in indices]

    # Dynamically create a new instance of the same class with additional attributes preserved
    # Extract existing attributes except 'collection'
    other_attrs = {k: v for k, v in vars(instance).items() if k != "collection"}
    return instance.__class__(collection=indexed_subset, **other_attrs)
