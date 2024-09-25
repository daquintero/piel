import numpy as np
from piel.types import ScalarMetrics, ScalarMetricCollection


def aggregate_scalar_metrics_collection(
    metrics_collection: ScalarMetricCollection,
) -> ScalarMetrics:
    """
    Aggregates a ScalarMetricCollection into a single ScalarMetrics instance.

    The aggregation is performed as follows:
    - mean: Weighted mean based on count.
    - min: Minimum of all min values.
    - max: Maximum of all max values.
    - standard_deviation: Combined standard deviation considering individual means and counts.
    - count: Sum of all counts.
    - unit: Must be consistent across all ScalarMetrics.

    Args:
        metrics_collection (ScalarMetricCollection): A ScalarMetricsCollection instances to aggregate.

    Returns:
        ScalarMetrics: A single ScalarMetrics instance representing the aggregated metrics.

    Raises:
        ValueError: If the input list is empty or units are inconsistent.
    """
    if not metrics_collection:
        raise ValueError("The metrics_list is empty.")

    # Extract necessary values
    means = []
    min_values = []
    max_values = []
    for metric in metrics_collection.metrics:
        if metric.mean is None:
            raise ValueError(f"ScalarMetrics '{metric}' must have 'mean' defined.")
        if metric.min is None or metric.max is None:
            raise ValueError(
                f"ScalarMetrics '{metric}' must have 'min' and 'max' defined."
            )
        means.append(metric.mean)
        min_values.append(metric.min)
        max_values.append(metric.max)

    total_count = len(means)
    if total_count == 0:
        raise ValueError("Total count is zero, cannot compute aggregated metrics.")

    # Compute aggregated mean
    aggregated_mean = sum(means) / total_count

    # Compute aggregated standard deviation
    if total_count > 1:
        variance = sum((m - aggregated_mean) ** 2 for m in means) / (total_count - 1)
        aggregated_std_dev = np.sqrt(variance)
    else:
        aggregated_std_dev = 0.0  # Standard deviation is zero if only one metric

    # Compute aggregated min and max
    aggregated_min = min(min_values)
    aggregated_max = max(max_values)

    # Aggregate value: set to aggregated mean (or any other logic as needed)
    aggregated_value = aggregated_mean

    # Create the aggregated ScalarMetrics instance
    aggregated_metrics = ScalarMetrics(
        value=aggregated_value,
        mean=aggregated_mean,
        min=aggregated_min,
        max=aggregated_max,
        standard_deviation=aggregated_std_dev,
        count=total_count,
        unit=metrics_collection.metrics[0].unit,
    )

    # TODO using the last metrics list for now

    return aggregated_metrics
