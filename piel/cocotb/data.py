"""
This file contains a range of functions used to read, plot and analyse cocotb simulations in a data-flow standard as suggested
"""
import functools
import pandas as pd
import bokeh as bh
from ..config import piel_path_types
from ..file_system import return_path, get_files_recursively_in_directory

__all__ = [
    "get_simulation_output_files",
    "get_simulation_output_files_from_design",
    "read_simulation_data",
    "simple_plot_simulation_data",
]

get_simulation_output_files = functools.partial(
    get_files_recursively_in_directory, path="./tb/out/", extension="csv"
)

""


def get_simulation_output_files_from_design(
    design_directory: piel_path_types,
    extension: str = "csv",
):
    """
    # TODO DOCS
    """
    design_directory = return_path(design_directory)
    output_files = get_files_recursively_in_directory(
        path=design_directory / "tb" / "out", extension=extension
    )
    return output_files


def read_simulation_data(file_path):
    """
    This function returns a Pandas dataframe that contains all the simulation data outputted from the simulation run.
    """
    file_path = return_path(file_path)
    simulation_data = pd.read_csv(file_path)
    return simulation_data


def simple_plot_simulation_data(simulation_data: pd.DataFrame):
    source = bh.models.ColumnDataSource(
        data=dict(time=simulation_data.time, signal=simulation_data.X)
    )

    p = bh.plotting.figure(
        height=300,
        width=800,
        tools="xpan",
        toolbar_location=None,
        x_axis_type="datetime",
        x_axis_location="above",
        background_fill_color="#efefef",
    )

    p.line("date", "close", source=source)
    p.yaxis.axis_label = "Price"

    select = bh.plotting.figure(
        title="Drag the middle and edges of the selection box to change the range above",
        height=130,
        width=800,
        y_range=p.y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    range_tool = bh.models.RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line("date", "close", source=source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    bh.plotting.show(bh.layouts.column(p, select))
