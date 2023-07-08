from ....file_system import return_path, get_files_recursively_in_directory

__all__ = [
    "filter_timing_sta_files",
    "filter_power_sta_files",
    "get_all_timing_sta_files",
    "get_all_power_sta_files",
]


def filter_timing_sta_files(file_list):
    """
    Filter the timing sta files from the list of files

    Args:
        file_list (list): List containing the file paths

    Returns:
        timing_sta_files (list): List containing the timing sta files
    """
    timing_sta_files = []
    for file_path in file_list:
        if file_path.endswith(("sta.rpt", "sta.min.rpt", "sta.max.rpt")):
            timing_sta_files.append(file_path)
    return timing_sta_files


def filter_power_sta_files(file_list):
    """
    Filter the power sta files from the list of files

    Args:
        file_list (list): List containing the file paths

    Returns:
        power_sta_files (list): List containing the power sta files
    """
    power_sta_files = []
    for file_path in file_list:
        if file_path.endswith(("power.rpt")):
            power_sta_files.append(file_path)
    return power_sta_files


def get_all_timing_sta_files(run_directory):
    """
    This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

    Args:
        run_directory (str, optional): The run directory to perform the analysis on. Defaults to None.

    Returns:
        timing_sta_files_list (list): List of all the .rpt files in the run directory.
    """
    run_directory = return_path(run_directory)
    all_rpt_files_list = get_files_recursively_in_directory(run_directory, "rpt")
    timing_sta_files_list = filter_timing_sta_files(all_rpt_files_list)
    return timing_sta_files_list


def get_all_power_sta_files(run_directory):
    """
    This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

    Args:
        run_directory (str, optional): The run directory to perform the analysis on. Defaults to None.

    Returns:
        power_sta_files_list (list): List of all the .rpt files in the run directory.
    """
    run_directory = return_path(run_directory)
    all_rpt_files_list = get_files_recursively_in_directory(run_directory, "rpt")
    power_sta_files_list = filter_power_sta_files(all_rpt_files_list)
    return power_sta_files_list
