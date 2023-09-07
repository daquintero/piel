from datetime import datetime
import pathlib
from piel.config import piel_path_types
from piel.file_system import return_path
from typing import Literal

__all__ = [
    "extract_datetime_from_path",
    "find_all_design_runs",
    "find_latest_design_run",
    "sort_design_runs",
]


def extract_datetime_from_path(run_path: pathlib.Path) -> str:
    """
    Extracts the datetime from a given `run_path` and returns it as a string.
    """
    run_str = run_path.name.replace("RUN_", "")
    if "." in run_str.split("_")[0]:
        date_format = "%Y.%m.%d_%H.%M.%S"
    else:
        date_format = "%Y-%m-%d_%H-%M-%S"
    return datetime.strptime(run_str, date_format)


def find_all_design_runs(
    design_directory: piel_path_types,
    run_name: str | None = None,
) -> list[pathlib.Path]:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

    If a `run_name` is specified, then the function will return the exact run if it exists. Otherwise, it will return the latest run

    Args:
        design_directory (piel_path_types): The path to the design directory
        run_name (str, optional): The name of the run to return. Defaults to None.

    Raises:
        ValueError: If the run_name is specified but not found in the design_directory

    Returns:
        list[pathlib.Path]: A list of pathlib.Path objects corresponding to the runs
    """
    design_directory = return_path(design_directory)
    runs_design_directory = design_directory / "runs"
    # Convert to path so that it can be found and compared within design_directory
    all_runs_list = list(runs_design_directory.iterdir())
    if run_name is not None:
        run_path = runs_design_directory / run_name
        # Return this exact run
        if run_path in all_runs_list:
            # Check that the run exists
            pass
        else:
            raise ValueError(
                "Run: " + str(run_path) + "not found in " + str(all_runs_list)
            )
    else:
        # Take the latest design run if it exists
        if len(all_runs_list) > 0:
            sorted_runs_per_version = sort_design_runs(all_runs_list)
        else:
            # If there are no runs
            raise ValueError(
                "No OpenLane design runs were found in: " + str(runs_design_directory)
            )
    return sorted_runs_per_version


def find_latest_design_run(
    design_directory: piel_path_types,
    run_name: str | None = None,
    version: Literal["v1", "v2"] | None = None,
) -> pathlib.Path:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

    If a `run_name` is specified, then the function will return the exact run if it exists. Otherwise, it will return the latest run.

    Args:
        design_directory (piel_path_types): The path to the design directory
        run_name (str, optional): The name of the run to return. Defaults to None.
        version (Literal["v1", "v2"], optional): The version of the run to return. Defaults to None.

    Raises:
        ValueError: If the run_name is specified but not found in the design_directory

    Returns:
        pathlib.Path: A pathlib.Path object corresponding to the latest run
    """
    latest_path = None
    latest_datetime = None

    sorted_runs_per_version = find_all_design_runs(
        design_directory=design_directory, run_name=run_name
    )

    # If a specific version is specified
    if version:
        filtered_paths = sorted_runs_per_version.get(version, [])
    else:
        filtered_paths = [
            path for sublist in sorted_runs_per_version.values() for path in sublist
        ]

    for path in filtered_paths:
        path_datetime = extract_datetime_from_path(path)
        if not latest_datetime or path_datetime > latest_datetime:
            latest_datetime = path_datetime
            latest_path = path

    return latest_path


def sort_design_runs(
    path_list: list[pathlib.Path],
) -> dict[str, list[pathlib.Path]]:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

    Args:
        path_list (list[pathlib.Path]): A list of pathlib.Path objects corresponding to the runs

    Returns:
        dict[str, list[pathlib.Path]]: A dictionary of sorted runs
    """
    # Initialize a dictionary to hold sorted PosixPaths categorized as v1 and v2
    sorted_paths = {"v2": list(), "v1": list()}

    for path in path_list:
        run_str = path.name.replace("RUN_", "")
        version = "v1" if "." in run_str.split("_")[0] else "v2"
        date_format = "%Y.%m.%d_%H.%M.%S" if version == "v1" else "%Y-%m-%d_%H-%M-%S"

        # Parsing datetime to make it sortable
        datetime_obj = datetime.strptime(run_str, date_format)

        sorted_paths[version].append((datetime_obj, path))

    # Sort the paths by datetime
    for version, paths in sorted_paths.items():
        sorted_paths[version] = [x[1] for x in sorted(paths)]

    # Return the sorted paths
    return sorted_paths
