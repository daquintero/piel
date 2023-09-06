import pathlib
from piel.config import piel_path_types
from piel.file_system import return_path

__all__ = [
    "find_all_design_runs",
    "find_latest_design_run",
]


def find_all_design_runs(
    design_directory: piel_path_types,
    run_name: str | None = None,
) -> list[pathlib.Path]:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

    These get

    They get sorted based on a reverse `list.sort()` method.

    # TODO docs
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
            all_runs_list.sort(reverse=True)
        else:
            # If there are no runs
            raise ValueError(
                "No OpenLane design runs were found in: " + str(runs_design_directory)
            )
    return all_runs_list


def find_latest_design_run(
    design_directory: piel_path_types,
    run_name: str | None = None,
) -> pathlib.Path:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

    They get sorted based on a reverse `list.sort()` method.

    # TODO docs
    """
    run_path = find_all_design_runs(
        design_directory=design_directory, run_name=run_name
    )[0]
    return run_path
