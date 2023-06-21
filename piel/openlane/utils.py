import pathlib
import json
from ..parametric import multi_parameter_sweep
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


def configure_parametric_designs(
    parameter_sweep_dictionary: dict,
    source_design_directory: str | pathlib.Path,
) -> list:
    """
    For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

    Args:
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        source_design_directory(str | pathlib.Path): Source design directory.

    Returns:
        configuration_sweep(list): List of configurations to sweep.
    """
    source_design_directory = return_path(source_design_directory)
    source_design_configuration_path = source_design_directory / "config.json"
    with open(source_design_configuration_path, "r") as config_json:
        source_configuration = json.load(config_json)
    configuration_sweep = multi_parameter_sweep(
        base_design_configuration=source_configuration,
        parameter_sweep_dictionary=parameter_sweep_dictionary,
    )
    return configuration_sweep


__all__ = [
    "configure_parametric_designs",
    "find_design_run",
]
