import numpy as np
import pandas as pd
from piel.types import SignalDCCollection
from .transfer.metrics import get_out_min_max, get_out_response_in_transition_range
from .transfer.power import get_power_metrics


def compile_dc_min_max_metrics_from_dc_collection(
    collections: list[SignalDCCollection],
    label_list: list[str],
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
        threshold_kwargs (dict, optional): Threshold kwargs for the transition transmission. Defaults to None.
        label_column_name (str, optional): How the label column should be called. Defaults to "label".
        debug (bool, optional): If True, raises exceptions during processing. Defaults to False.
        **kwargs: Additional keyword arguments for pd.DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the compiled metrics with combined min-max columns.
    """
    if not (len(collections) == len(label_list)):
        raise ValueError("Length of collections and label_list must be equal.")

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
            power_max = power_metrics.max / 1e-3  # Convert to mW
            power_delta = (
                power_metrics.max - power_metrics.min
            ) / 1e-3  # Convert to mW

            # Format metrics to three decimal places
            formatted_vin = f"{min_vin:.3f}-{max_vin:.3f}"
            formatted_vout = f"{vout_min:.3f}-{vout_max:.3f}"
            formatted_power_max = f"{power_max:.3f}"
            formatted_power_delta = f"{power_delta:.3f}"

            # Append to data
            data.append(
                {
                    label_column_name: label,
                    r"$V_{out}$ $V$": formatted_vout,
                    r"$V_{tr,in}$ $V$": formatted_vin,
                    r"$P_{dd,max}$ $mW$": formatted_power_max,
                    r"$\Delta P_{dd}$ $mW$": formatted_power_delta,
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
                    r"$V_{out}$ $V$": np.nan,
                    r"$V_{tr,in}$ $V$": np.nan,
                    r"$P_{dd,max}$ $mW$": np.nan,
                    r"$\Delta P_{dd}$ $mW$": np.nan,
                }
            )
            if debug:
                raise e

    # Create DataFrame
    final_df = pd.DataFrame(data, **kwargs)

    return final_df


def compile_dc_transition_metrics_from_dc_collection(
    collections: list[SignalDCCollection],
    label_list: list[str],
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
        threshold_kwargs (dict, optional): Threshold kwargs for the transition transmission. Defaults to None.
        label_column_name (str, optional): How the label column should be called. Defaults to "label".
        debug (bool, optional): If True, raises exceptions during processing. Defaults to False.
        **kwargs: Additional keyword arguments for pd.DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the compiled metrics with combined min-max columns.
    """
    if not (len(collections) == len(label_list)):
        raise ValueError("Length of collections and label_list must be equal.")

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

            # Extract metrics
            min_vin = vin_metrics.min
            max_vin = vin_metrics.max
            vout_min = vout_metrics.min
            vout_max = vout_metrics.max

            # Format metrics to three decimal places
            formatted_vin = f"{min_vin:.3f}-{max_vin:.3f}"
            formatted_vout = f"{vout_min:.3f}-{vout_max:.3f}"

            # Append to data
            data.append(
                {
                    label_column_name: label,
                    r"$V_{out}$ $V$": formatted_vout,
                    r"$V_{tr,in}$ $V$": formatted_vin,
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
                    r"$V_{out}$ $V$": np.nan,
                    r"$V_{tr,in}$ $V$": np.nan,
                }
            )
            if debug:
                raise e

    # Create DataFrame
    final_df = pd.DataFrame(data, **kwargs)

    return final_df
