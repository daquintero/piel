import numpy as np
import pandas as pd
from typing import List
from piel.types import SignalDCCollection
from .transfer.metrics import get_out_min_max, get_out_response_in_transition_range
from .transfer.power import get_power_metrics


def compile_dc_min_max_metrics_from_dc_collection(
    collections: List[SignalDCCollection],
    label_list: List[str],
    label_column_name: str = "label",
    threshold_kwargs: dict = None,
    debug: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Compiles DC analysis metrics from a list of SignalDCCollection instances into a final DataFrame.

    Args:
        collections (List[SignalDCCollection]): List of SignalDCCollection instances to analyze.
        label_list (List[str]): List of labels corresponding to each SignalDCCollection.
        threshold_kwargs(dict): Threshold kwargs for the transition transmission
        label_column_name(str): How the label column should be called

    Returns:
        pd.DataFrame: A DataFrame containing the compiled metrics.
    """
    if not (len(collections) == len(label_list)):
        raise ValueError("Length of collections, labels must be equal.")

    if threshold_kwargs is None:
        threshold_kwargs = {}

    data = []

    for idx, (collection, label) in enumerate(zip(collections, label_list)):
        try:
            # Apply analysis functions
            vout_metrics = get_out_min_max(collection, **threshold_kwargs)
            vin_metrics = get_out_response_in_transition_range(
                collection, **threshold_kwargs
            )
            power_metrics = get_power_metrics(collection, **threshold_kwargs)

            # Extract metrics
            min_vin = vin_metrics.min
            max_vin = vin_metrics.max
            vout_min = vout_metrics.min
            vout_max = vout_metrics.max
            power_max = power_metrics.max / 1e-3
            power_delta = (power_metrics.max - power_metrics.min) / 1e-3

            # Append to data
            data.append(
                {
                    label_column_name: label,
                    r"$V_{out, min}$ $V$": vout_min,
                    r"$V_{out, max}$ $V$": vout_max,
                    r"$V_{tr,in, min}$ $V$": min_vin,
                    r"$V_{tr,in, max}$ $V$": max_vin,
                    r"$P_{dd,max}$ $mW$": power_max,
                    r"$\Delta P_{dd}$ $mW$": power_delta,
                }
            )

        except Exception as e:
            print(
                f"Error processing collection at index {idx} with label '{label}': {e}"
            )
            # Optionally, append NaNs or skip
            data.append(
                {
                    label_column_name: label,
                    r"$V_{out, min}$ $V$": np.nan,
                    r"$V_{out, max}$ $V$": np.nan,
                    r"$V_{tr,in, min}$ $V$": np.nan,
                    r"$V_{tr,in, max}$ $V$": np.nan,
                    r"$P_{dd,max}$ $mW$": np.nan,
                    r"$\Delta P_{dd}$ $mW$": np.nan,
                }
            )
            if debug:
                raise e

    # Create DataFrame
    final_df = pd.DataFrame(data, **kwargs).apply(lambda x: round(x, 3))

    return final_df
