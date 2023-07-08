import pathlib
from piel.config import piel_path_types
from piel.file_system import return_path

__all__ = [
    "find_design_run",
]


def find_design_run(
    design_directory: piel_path_types,
    run_name: str | None = None,
) -> pathlib.Path:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

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
            latest_run = all_runs_list[0]
            run_path = latest_run
        else:
            # If there are no runs
            raise ValueError(
                "No OpenLane design runs were found in: " + str(runs_design_directory)
            )
    return run_path
