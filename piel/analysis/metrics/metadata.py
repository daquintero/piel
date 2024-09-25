from piel.types import ScalarMetricCollection


def rename_metrics_collection(
    collection: ScalarMetricCollection, new_names: list[str]
) -> ScalarMetricCollection:
    """
    Renames each metric in the provided ScalarMetricCollection with the corresponding name from new_names.

    Args:
        collection (ScalarMetricCollection): The original metric collection.
        new_names (List[str]): A list of new names for the metrics.

    Returns:
        ScalarMetricCollection: A new metric collection with renamed metrics.

    Raises:
        ValueError: If the number of new names does not match the number of metrics.
    """
    if len(new_names) != len(collection.metrics):
        raise ValueError(
            f"Number of new names ({len(new_names)}) does not match "
            f"the number of metrics in the collection ({len(collection.metrics)})."
        )

    # Create a new list of metrics with updated names
    updated_metrics = []
    for metric, new_name in zip(collection.metrics, new_names):
        updated_metric = metric.copy(update={"name": new_name})
        updated_metrics.append(updated_metric)

    # Return a new ScalarMetricCollection with the updated metrics
    return collection.copy(update={"metrics": updated_metrics})
