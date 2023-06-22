"""
TODO these functions do not currently work.
"""

import glob
import os
from .openlane_opensta_v1 import run_parser


def get_all_rpt_files(run_directory=None):
    """
    This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

    Args:
        run_directory (str, optional): The run directory to perform the analysis on. Defaults to None.

    Returns:
        all_rpt_files_list (list): List of all the .rpt files in the run directory.
        timing_sta_files_list (list): List of all the .rpt files in the run directory.
        power_sta_files_list (list): List of all the .rpt files in the run directory.
    """
    all_rpt_files_list = []
    timing_sta_files_list = []
    power_sta_files_list = []
    PATH = os.path.dirname(os.getcwd())
    for x in os.walk(PATH):
        for file_path in glob.glob(os.path.join(x[0], "*.rpt")):
            all_rpt_files_list.append(file_path)
            if file_path.endswith(
                (
                    "sta.rpt",
                    "sta.min.rpt",
                    "sta.max.rpt",
                )
            ):
                timing_sta_files_list.append(file_path)
            if file_path.endswith(("power.rpt")):
                power_sta_files_list.append(file_path)
    return all_rpt_files_list, timing_sta_files_list, power_sta_files_list


def extract_metrics_timing(timing_sta_files_list):
    """
    For every file in the sta timing file, extract the propagation delay and save the file meta data into a dictionary.

    Args:
        timing_sta_files_list (list): List of all the .rpt files in the run directory.

    Returns:
        timing_metrics_list (list): List of dictionaries containing the file meta data and the propagation delay.
    """
    timing_metrics_list = []
    for file in timing_sta_files_list:
        file_directory_data = os.path.normpath(file).split(os.path.sep)
        frame_timing_data, propagation_delay = run_parser(file_address=file)
        metrics = {
            "file_name": file_directory_data[-1],
            "flow_step_name": file_directory_data[-2],
            "propagation_delay": propagation_delay,
        }
        timing_metrics_list.append(metrics)
    return timing_metrics_list


def run_analysis(run_directory):
    """
    This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

    Args:
        run_directory (str, optional): The run directory to perform the analysis on. Defaults to None.

    Returns:
        timing_metrics_list (list): List of dictionaries containing the file meta data and the propagation delay.
    """
    all_rpt_files_list, timing_sta_files_list, power_sta_files_list = get_all_rpt_files(
        run_directory
    )
    timing_metrics_list = extract_metrics_timing(timing_sta_files_list)
    return timing_metrics_list
