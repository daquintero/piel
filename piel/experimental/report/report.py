from ...types import PathTypes
from ...file_system import create_new_directory
from ..measurements.data.extract import load_experiment_data_from_directory
from .plots import create_plots_from_experiment_data
from ..text import write_experiment_top_markdown, append_image_path_list_to_markdown


def create_report():
    """
    This functionality is used to create a report of a given directory containing the `ExperimentCollection`
    and the `Experiments` themselves.

    This uses the metadata of the `Experiment`s and the corresponding `ExperimentData`
    to generate the corresponding images.
    """
    pass


def create_report_from_experiment_directory(
    experiment_directory: PathTypes,
    plot_output_directory: PathTypes = None,
    report_readme_path: PathTypes = None,
    load_data_kwargs: dict = None,
    plot_kwargs: dict = None,
    **kwargs,
):
    """
    First we need to extract the `ExperimentData` from the directory.
    Then we can generate the report from the `ExperimentData`.

    """
    if plot_output_directory is None:
        # Create an image output directory as required
        plot_output_directory = experiment_directory / "img"
        create_new_directory(plot_output_directory)

    if report_readme_path is None:
        # Create a report README.md file
        main_readme_path = experiment_directory / "README.md"
        report_readme_path = experiment_directory / "REPORT.md"

    if load_data_kwargs is None:
        load_data_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    # We load the experiment data from the metadata
    experiment_data = load_experiment_data_from_directory(
        experiment_directory=experiment_directory, **load_data_kwargs
    )

    # Now we need to generate all plots accordingly
    plots, plots_paths = create_plots_from_experiment_data(
        experiment_data=experiment_data,
        plot_output_directory=plot_output_directory,
        experiment_directory=experiment_directory,
        **plot_kwargs,
    )

    # Now we need to generate the new experiment README.md
    # We want to append the parametric information of the experiment here.
    # Note that we want to append the main README with this data.
    write_experiment_top_markdown(
        experiment=experiment_data.experiment,
        experiment_directory=experiment_directory,
        target_markdown_file=main_readme_path,
    )
    write_experiment_top_markdown(
        experiment=experiment_data.experiment,
        experiment_directory=experiment_directory,
        target_markdown_file=report_readme_path,
    )

    # Now we need to append all plots to the READMEs
    append_image_path_list_to_markdown(
        image_path_list=plots_paths, markdown_file=main_readme_path
    )
    append_image_path_list_to_markdown(
        image_path_list=plots_paths, markdown_file=report_readme_path
    )

    print(f"README.md updated at: {main_readme_path}")
    print(f"REPORT.md written to: {report_readme_path}")
