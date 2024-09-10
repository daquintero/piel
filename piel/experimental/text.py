import pandas as pd

from ..types import PathTypes
from piel.types.experimental import Experiment
from ..file_system import (
    return_path,
    write_file,
    read_json,
)
from ..visual import dictionary_to_markdown_str


def append_image_path_list_to_markdown(
    image_path_list: list[str],
    markdown_file: str,
):
    """
    Appends a list of image paths to a markdown file.

    Parameters:
        image_paths (list): A list of image file paths to be added to the markdown file.
        markdown_file (str): The path to the markdown file.
    """
    markdown_file = return_path(markdown_file)

    with open(markdown_file, "a") as md_file:
        for image_path in image_path_list:
            image_path = return_path(image_path)
            assert image_path.exists()

            # Calculate the relative path from the markdown file to the image
            relative_image_path = image_path.relative_to(markdown_file.parent)

            # Assuming you want to append the image with markdown syntax
            md_file.write(f"\n\n![Image]({str(relative_image_path)})\n\n")


def write_schema_markdown(schema_json_file: PathTypes, target_markdown_file: PathTypes):
    """
    This function writes the schema markdown file for the experiment configuration. This schema markdown file should
    contain all the required information to understand the experiment configuration. This should include all the
    experiment instances and their corresponding configurations.

    """
    schema_json_file = return_path(schema_json_file)
    schema_json_file = read_json(schema_json_file)
    schema_markdown = dictionary_to_markdown_str(schema_json_file)
    schema_markdown = "\n\n## Schema \n" + schema_markdown
    write_file(
        target_markdown_file.parent,
        schema_markdown,
        target_markdown_file.name,
        append=True,
    )


def write_experiment_top_markdown(
    experiment: Experiment,
    experiment_directory: PathTypes,
    target_markdown_file: PathTypes = None,
):
    if target_markdown_file is None:
        target_markdown_file = experiment_directory / "README.md"

    # Experiment Top Level
    markdown_top_text = (
        "# "
        + str(experiment.name)
        + "\n\n**Goal**: "
        + str(experiment.goal)
        + "\n\n## Experiment Parameters \n\n"
    )

    # Write the configuration markdown file and README accordingly. The README gets overwritten based on the data analysis.
    write_file(
        target_markdown_file.parent,
        markdown_top_text,
        target_markdown_file.name,
    )

    # Construct the iteration parameters table and save iteration table to markdown
    iteration_parameters_table = pd.DataFrame(experiment.parameters_list)
    iteration_parameters_table.to_markdown(target_markdown_file, mode="a")
    return None
