import numpy as np
from piel.types import MultiDataTimeSignal, ScalarMetrics, EdgeTransitionAnalysisTypes
from piel.types.units import s
from piel.analysis.metrics import aggregate_scalar_metrics_list


def extract_mean_metrics_list(
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


def extract_peak_to_peak_metrics_list(
    multi_data_time_signal: MultiDataTimeSignal,
) -> list[ScalarMetrics]:
    """
    Extracts peak-to-peak metrics from a collection of signals. The peak-to-peak value is defined as the
    difference between the maximum and minimum values of the signal.

    Args:
        multi_data_time_signal (MultiDataTimeSignal): A collection of time signals to analyze.

    Returns:
        List[ScalarMetrics]: A list of ScalarMetrics instances containing the peak-to-peak values
                             for each signal.

    Raises:
        ValueError: If the input list is empty or any signal has an empty data array.
    """
    if not multi_data_time_signal:
        raise ValueError("The multi_data_time_signal list is empty.")

    metrics_list = []

    for signal in multi_data_time_signal:
        if not signal.data:
            raise ValueError(f"Signal '{signal.data_name}' has an empty data array.")

        data_array = np.array(signal.data)

        min_val = float(np.min(data_array))
        max_val = float(np.max(data_array))
        peak_to_peak = max_val - min_val

        scalar_metric = ScalarMetrics(
            value=peak_to_peak,  # Using peak-to-peak as the primary value
            mean=peak_to_peak,  # Mean is not applicable for peak-to-peak
            min=peak_to_peak,  # Min is already represented in peak-to-peak
            max=peak_to_peak,  # Max is already represented in peak-to-peak
            standard_deviation=None,  # Not applicable
            count=None,  # Not applicable
            unit=s,  # Adjust the unit if peak-to-peak has different units
        )

        metrics_list.append(scalar_metric)

    return metrics_list


def extract_statistical_metrics(
    multi_data_time_signal: MultiDataTimeSignal,
    analysis_type: EdgeTransitionAnalysisTypes = "peak_to_peak",
) -> ScalarMetrics:
    """
    Extracts scalar metrics from a collection of rising edge signals.

    Args:
        multi_data_time_signal (List[DataTimeSignalData]): A list of rising edge signals.
        analysis_type (piel.types.EdgeTransitionAnalysisTypes): The type of analysis to perform.

    Returns:
        ScalarMetrics: Aggregated ScalarMetrics instance containing the extracted metrics.

    """
    if analysis_type == "mean":
        metrics_list = extract_mean_metrics_list(multi_data_time_signal)
    elif analysis_type == "peak_to_peak":
        metrics_list = extract_peak_to_peak_metrics_list(multi_data_time_signal)
    aggregate_metrics = aggregate_scalar_metrics_list(metrics_list)
    return aggregate_metrics
