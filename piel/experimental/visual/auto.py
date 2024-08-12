from ...types import PathTypes
from ..types import MeasurementDataTypes, MeasurementDataCollectionTypes, ExperimentData
from .map import (
    measurement_data_collection_to_plot_map,
    measurement_data_to_plot_map,
    measurement_data_collection_to_plot_suffix_map,
)


def auto_plot_from_measurement_data(
    measurement_data: MeasurementDataTypes,
    **kwargs,
) -> list[tuple, PathTypes]:
    """
    This function will automatically plot the data from the `MeasurementData` object provided.
    If there are more than one set of relevant plots for a given `MeasurementData`,
    it will generate a list of figures accordingly.
    """
    plots = []
    plot_methods = measurement_data_to_plot_map[measurement_data.type]
    for plot_method_i in plot_methods:
        plot_i = plot_method_i(measurement_data, **kwargs)
        plots.append(plot_i)

    return plots


def auto_plot_from_measurement_data_collection(
    measurement_data_collection: MeasurementDataCollectionTypes,
    plot_output_directory: PathTypes = None,
    measurement_data_collection_to_plot_map: dict = measurement_data_collection_to_plot_map,
    measurement_data_collection_to_plot_prefix_map: dict = measurement_data_collection_to_plot_suffix_map,
    **kwargs,
) -> tuple[list[tuple], list[PathTypes]]:
    """
    This function will automatically plot the data from the `MeasurementDataCollection` provided.
    If there are more than one set of relevant plots for a given `MeasurementData`,
    it will generate a list of figures accordingly.
    """
    plots = []
    plot_path_list = []

    # This creates the mapping between measureemnt collection and the corresponding plots
    plot_methods = measurement_data_collection_to_plot_map[
        measurement_data_collection.type
    ]

    plot_prefix = measurement_data_collection_to_plot_prefix_map[
        measurement_data_collection.type
    ]

    i = 0
    # We iterate through the corresponding plotting methods and generate the plots, and save them to the parent directory.
    for plot_method_i in plot_methods:
        plot_file_i = (
            plot_output_directory
            / f"{measurement_data_collection.name}_{plot_prefix[i]}.png"
        )
        plot_i = plot_method_i(measurement_data_collection, path=plot_file_i, **kwargs)
        plots.append(plot_i)
        plot_path_list.append(plot_file_i)
        i += 1

    return plots, plot_path_list


def auto_plot_from_experiment_data(
    experiment_data: ExperimentData, plot_output_directory: PathTypes = None, **kwargs
) -> tuple[list[tuple], list[PathTypes]]:
    """
    This function will automatically plot the data from the `ExperimentData` object provided.
    """
    plots, plot_path_list = auto_plot_from_measurement_data_collection(
        measurement_data_collection=experiment_data.data,
        plot_output_directory=plot_output_directory,
        **kwargs,
    )
    return plots, plot_path_list
