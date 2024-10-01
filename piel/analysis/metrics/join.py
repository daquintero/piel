from piel.types import ScalarMetricCollection


def concatenate_metrics_collection(
    metrics_collection_list: list[ScalarMetricCollection], **kwargs
) -> ScalarMetricCollection:
    """
    Concatenates multiple ScalarMetricCollection instances into a single ScalarMetricCollection.

    Args:
        metrics_collection_list (List[ScalarMetricCollection]): List of ScalarMetricCollection instances to concatenate.

    Returns:
        ScalarMetricCollection: A new ScalarMetricCollection containing all metrics from the input collections.

    Raises:
        ValueError: If the input list is empty.
    """
    if not metrics_collection_list:
        raise ValueError(
            "The metrics_collection_list is empty. Provide at least one ScalarMetricCollection."
        )

    total_metrics_list = list()

    for collection in metrics_collection_list:
        if not isinstance(collection, ScalarMetricCollection):
            raise TypeError(
                f"Collection {collection} is the issue. All items in metrics_collection_list must be instances of ScalarMetricCollection."
            )
        total_metrics_list.extend(collection.metrics)

    return ScalarMetricCollection(metrics=total_metrics_list, **kwargs)
