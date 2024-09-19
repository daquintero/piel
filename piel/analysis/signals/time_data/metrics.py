import numpy as np
from piel.types import MultiDataTimeSignal, ScalarMetrics
from piel.types.units import s
from piel.analysis.metrics import aggregate_scalar_metrics_list


def extract_rising_edge_metrics_list(
    multi_data_time_signal: MultiDataTimeSignal,
) -> list[ScalarMetrics]:
    """
    Extracts scalar metrics from a collection of rising edge signals. Standard deviation is not calculated as this just
    computes individual metrics list.

    Args:
        multi_data_time_signal (List[DataTimeSignalData]): A list of rising edge signals.

    Returns:
        List[ScalarMetrics]: A list of ScalarMetrics instances containing the extracted metrics.
    """
    if not multi_data_time_signal:
        raise ValueError("The multi_signal list is empty.")

    metrics_list = []

    for signal in multi_data_time_signal:
        if not signal.data:
            raise ValueError(f"Signal '{signal.data_name}' has an empty data array.")

        data_array = np.array(signal.data)

        mean_val = float(np.mean(data_array))
        min_val = float(np.min(data_array))
        max_val = float(np.max(data_array))
        std_dev = None
        count = None

        # Assuming 'value' is the mean; adjust if different meaning is intended
        scalar_metric = ScalarMetrics(
            value=mean_val,
            mean=mean_val,
            min=min_val,
            max=max_val,
            standard_deviation=std_dev,
            count=count,
            unit=s,
        )

        metrics_list.append(scalar_metric)

    return metrics_list


def extract_rising_edge_statistical_metrics(
    multi_data_time_signal: MultiDataTimeSignal,
) -> ScalarMetrics:
    """
    Extracts scalar metrics from a collection of rising edge signals.

    Args:
        multi_data_time_signal (List[DataTimeSignalData]): A list of rising edge signals.

    Returns:
        ScalarMetrics: Aggregated ScalarMetrics instance containing the extracted metrics.

    """
    metrics_list = extract_rising_edge_metrics_list(multi_data_time_signal)
    aggregate_metrics = aggregate_scalar_metrics_list(metrics_list)
    return aggregate_metrics
