"""
This module provides functions to read, plot, and analyze Cocotb simulation files. It supports reading simulation
output files, converting them into Pandas dataframes, and plotting the files using Bokeh for interactive visualization.
# TODO: Implement the logic for processing different signal measurement.
"""

import functools
import pandas as pd
from piel.types import PathTypes
from piel.file_system import return_path, get_files_recursively_in_directory

# Partial function to get all CSV files from the 'tb/out' directory.
get_simulation_output_files = functools.partial(
    get_files_recursively_in_directory, path="./tb/out/", extension="csv"
)


def get_simulation_output_files_from_design(
    design_directory: PathTypes,
    extension: str = "csv",
) -> list:
    """
    Returns a list of all simulation output files in the specified design directory.

    Args:
        design_directory (PathTypes): The path to the design directory.
        extension (str, optional): The file extension to filter by. Defaults to "csv".

    Returns:
        list: A list of paths to the simulation output files in the design directory.

    Examples:
        >>> get_simulation_output_files_from_design("/path/to/design")
        [PosixPath('/path/to/design/tb/out/output1.csv'), PosixPath('/path/to/design/tb/out/output2.csv')]
    """
    design_directory = return_path(design_directory)
    output_files = get_files_recursively_in_directory(
        path=design_directory / "tb" / "out", extension=extension
    )
    return output_files


def read_simulation_data(file_path: PathTypes, *args, **kwargs) -> pd.DataFrame:
    """
    Reads simulation files from a specified file into a Pandas dataframe.

    Args:
        file_path (PathTypes): The path to the simulation files file.

    Returns:
        pd.DataFrame: The simulation files as a Pandas dataframe.

    Examples:
        >>> read_simulation_data("/path/to/simulation/output.csv")
        # Returns a dataframe with the contents of the CSV file.
    """
    file_path = return_path(file_path)
    simulation_data = pd.read_csv(
        file_path, dtype=str, encoding="utf-8", *args, **kwargs
    )
    return simulation_data


def simple_plot_simulation_data(simulation_data: pd.DataFrame):
    """
    Plots simulation files using Bokeh for interactive visualization.

    Args:
        simulation_data (pd.DataFrame): The simulation files to plot, containing columns 't' for time and 'x' for signal values.

    Returns:
        None: Displays an interactive plot.

    Examples:
        >>> files = pd.DataFrame({"t": [0, 1, 2, 3], "x": [0, 1, 0, 1]})
        >>> simple_plot_simulation_data(files)
        # Displays an interactive Bokeh plot.
    """
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure, show
    from bokeh.layouts import column

    source = ColumnDataSource(
        data=dict(time=simulation_data["t"], signal=simulation_data["x"])
    )

    p = figure(
        height=300,
        width=800,
        tools="xpan",
        toolbar_location=None,
        x_axis_location="above",
        background_fill_color="#efefef",
        title="Simulation Signal over Time",
    )

    p.line("time", "signal", source=source, line_width=2)
    p.yaxis.axis_label = "Signal"

    select = figure(
        title="Drag to change the range above",
        height=130,
        width=800,
        y_range=p.y_range,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    select.line("time", "signal", source=source, line_width=1)
    select.ygrid.grid_line_color = None

    show(column(p, select))
