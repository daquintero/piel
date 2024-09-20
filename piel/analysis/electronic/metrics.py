from piel.types import RFAmplifierCollection, ComponentMetrics, ScalarMetrics
import pandas as pd


def compose_amplifier_collection_performance_dataframe(
    amplifier_collection: RFAmplifierCollection, desired_metrics: list[str]
) -> pd.DataFrame:
    """
    Composes performance parameters of amplifiers into a pandas DataFrame,
    handling multiple metrics per component.

    Args:
        amplifier_collection (RFAmplifierCollection): The collection of RF amplifiers.
        desired_metrics (List[str]): List of metric names to include in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the specified metrics for each amplifier and metrics instance.
    """
    records = []

    for comp_idx, component in enumerate(amplifier_collection.components, start=1):
        metrics_list: list[ComponentMetrics] = getattr(component, "metrics", [])

        if not metrics_list:
            # If no metrics are present, fill with None
            record = {"Component_ID": comp_idx, "Metrics_Instance": None}
            for metric in desired_metrics:
                record[metric] = None
            records.append(record)
            continue

        for metric_idx, metrics in enumerate(metrics_list, start=1):
            record = {"Component_ID": comp_idx, "Metrics_Instance": metric_idx}

            for metric in desired_metrics:
                # Use getattr to safely access the metric; default to None if not present
                metric_obj = getattr(metrics, metric, None)

                if isinstance(metric_obj, ScalarMetrics):
                    # Assuming ScalarMetrics has 'min' and 'max' attributes
                    # You may need to adjust based on actual implementation
                    record[f"{metric}_min"] = getattr(metric_obj, "min", None)
                    record[f"{metric}_max"] = getattr(metric_obj, "max", None)
                else:
                    # For non-ScalarMetrics (e.g., strings), assign directly
                    record[metric] = metric_obj

            records.append(record)

    # Create DataFrame from records
    df = pd.DataFrame(records)

    return df
