import pathlib
from ..file_system import return_path


def find_design_run(
    design_directory: str | pathlib.Path,
    run_name: str | None = None,
) -> str:
    """
    For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

    They get sorted based on a reverse `list.sort()` method.
    """
    design_directory = return_path(design_directory)
    runs_design_directory = design_directory / "runs"
    all_runs_list = list(runs_design_directory.iterdir())
    if run_name is not None:
        all_runs_list.sort(reverse=True)
        latest_run = all_runs_list[0]
    elif run_name in all_runs_list:
        latest_run = run_name
    else:
        raise ValueError("Run: " + run_name + "not found in " + str(all_runs_list))
    return runs_design_directory / latest_run


__all__ = [
    "find_design_run",
]
