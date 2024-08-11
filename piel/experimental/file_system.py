"""
This file contains all functionality required to verify the corresponding experimental file structure and the
mapping between experimental data, configuration ids, and the file structure accordingly. The goal would be to create
a direct mapping between an operating setup configuration, or experiment.
"""

from ..types import PathTypes
from .types import Experiment
from ..file_system import (
    create_new_directory,
    return_path,
    write_model_to_json,
)
from .text import write_schema_markdown, write_experiment_top_markdown


def construct_experiment_directories(
    experiment: Experiment,
    parent_directory: PathTypes,
    construct_directory: bool = True,
) -> PathTypes:
    """
    This function constructs the directories of the experiment configuration. It iterates through the experiment
    instances. It checks that each of these instances is unique. Each experiment.experiment_instance should have a
    unique name. This directory name is an enumerated integer. This enumerated integer is the index of the experiment
    instance in the experiment.experiment_instances tuple. This function should be able to create a new experiment
    configuration from scratch.

    A parent directory is defined in which to create the experiment directories. The experiment directory is created
    in the parent directory. The experiment directory contains the experiment.json file. The experiment.json file
    contains the experiment configuration. Note that the experiment.json file should be recursive.
    The experiment directory also contains the experiment instances. The
    experiment instances are directories. Each experiment instance directory contains the instance.json file. The
    instance.json file contains the experiment instance configuration. This is as flat as the directory structure gets.

    The instance directory will contain the data files alongside all of this metadata information.
     The data files are at the top level of the instance directory, and should not have subdirectories.

    The data files are manually generated from the corresponding measurements specified in the instance.json file.
    These will be added after wards from this directory structure creation.

    This schema is used to generate a README file for the experiment configuration. This README file should contain
    all the information about the experiment configuration. This includes the experiment instances and their
    corresponding configurations.

    Parameters
    ----------
    experiment : Experiment
        The experiment configuration to create the directories for.

    parent_directory : PathTypes
        The parent directory to create the experiment directory in.

    Returns
    -------
    PathTypes
        The path to the experiment directory.

    """
    parent_directory = return_path(parent_directory)
    # Create the experiment directory
    experiment_directory = parent_directory / experiment.name

    if (experiment_directory.exists()) and not construct_directory:
        input(
            "The directory already exists. Press enter to continue and overwrite all the json and configuration "
            "files."
        )
    else:
        create_new_directory(experiment_directory)

    # Create the experiment.json file
    experiment_json_path = experiment_directory / "experiment.json"
    write_model_to_json(experiment, experiment_json_path)

    # Create the experiment README.md file
    experiment_markdown_path = experiment_directory / "README.md"
    setup_experiment_markdown_path = experiment_directory / "SETUP.md"
    write_experiment_top_markdown(
        experiment, experiment_directory, experiment_markdown_path
    )
    write_experiment_top_markdown(
        experiment, experiment_directory, setup_experiment_markdown_path
    )

    # Append schema to markdown
    write_schema_markdown(experiment_json_path, experiment_markdown_path)
    write_schema_markdown(experiment_json_path, setup_experiment_markdown_path)

    # Create the experiment instances
    for index, experiment_instance in enumerate(experiment.experiment_instances):
        # Create the experiment instance directory
        experiment_instance_directory = experiment_directory / str(index)
        create_new_directory(experiment_instance_directory)

        # Create the instance.json file
        instance_json_path = experiment_instance_directory / "instance.json"
        instance_markdown_path = experiment_instance_directory / "README.md"
        write_model_to_json(experiment_instance, instance_json_path)
        write_schema_markdown(instance_json_path, instance_markdown_path)

    print(f"Experiment directory created at {experiment_directory}")
    return experiment_directory


def construct_experiment_structure(experiment: Experiment, parent_directory: PathTypes):
    """
    The goal of this function is to construct both the directories and the json file which defines both the experiment
    and all the experiment instances. It should be able to create a new experiment configuration from scratch.

    """
