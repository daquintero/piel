import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
import numpy as np


def extract_figure_bottom_bbox(axs):
    """
    Extracts the bounding box that covers the bottom range of the figure, considering multiple subplots.
    Handles both 1D and 2D arrays of axes.
    """
    # Initialize bounding box to the extremes
    x0, y0, x1, y1 = 1, 1, 0, 0

    # Flatten the axs array if it's 2D or higher, otherwise treat it as a list
    if isinstance(axs, np.ndarray):
        axs = axs.ravel()
    else:
        axs = [axs]

    for ax in axs:
        if ax is None:  # In case of empty subplots
            continue
        bbox = ax.get_position()
        # Update the bbox boundaries based on each subplot
        x0 = min(x0, bbox.x0)
        y0 = min(y0, bbox.y0)
        x1 = max(x1, bbox.x1)
        y1 = max(y1, bbox.y1)

    # Return the combined bounding box
    return [x0, y0, x1 - x0, y1 - y0]


def create_axes_parameters_table_overlay(
    fig,
    axs: list,
    parameters_list: list,
    font_family="Roboto",
    header_font_weight="bold",
    cell_font_size=10,
    cell_font_weight="normal",
):
    """
    This function takes in the parameter_list and a figure, axes list, to return a figure and axes with an attached
    parameter table and relevant colors and line styles. The figure must already have the axes plotted and ready to extract the
    relevant colors, line styles, and parameters from it accordingly.

    This function is particularly useful if the parametric sweep is overlaid in multiple lines of the same axes.
    """
    # Get the combined bounding box of the bottom axes
    bbox1 = extract_figure_bottom_bbox(axs)

    # Calculate a new position for the table axes (e.g., below the existing axes)
    new_position = [bbox1[0], bbox1[1] - bbox1[3] * 0.4, bbox1[2], bbox1[3] * 0.4]

    # Add a new set of axes in the calculated position for the table
    table_ax = fig.add_axes(new_position)

    # Extract the lines from the original axes (last subplot)
    lines_list = axs.ravel()[-1].get_lines()

    # Extract colors and line styles from the lines
    colors = [line.get_color() for line in lines_list]
    line_styles = [line.get_linestyle() for line in lines_list]

    # Prepare the data for the table, including color and line style
    plot_line_data = []
    for parameters_i, _, _ in zip(parameters_list, colors, line_styles):
        plot_line_entry = (
            parameters_i.copy()
        )  # Copy the original dictionary to avoid modifying it
        plot_line_entry["Color & Style"] = ""  # This will be replaced by the drawn line
        plot_line_data.append(plot_line_entry)

    # Convert to DataFrame
    axes_dataframe = pd.DataFrame(plot_line_data)

    # Remove the new axes border and ticks
    table_ax.axis("off")

    # Create a custom table
    table = table_ax.table(
        cellText=axes_dataframe.values,
        colLabels=axes_dataframe.columns,
        loc="center",
        cellLoc="center",
    )

    # Ensure that the table is drawn
    plt.draw()

    # Set font properties
    font_properties = fm.FontProperties(
        family=font_family, size=cell_font_size, weight=cell_font_weight
    )
    header_font_properties = fm.FontProperties(
        family=font_family, size=cell_font_size, weight=header_font_weight
    )

    # Set font for header
    for (i, _), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(fontproperties=header_font_properties)
        else:  # All other cells
            cell.set_text_props(fontproperties=font_properties)

    # Draw the line style and color in each corresponding cell
    for i, (color, line_style) in enumerate(zip(colors, line_styles)):
        cell = table[(i + 1, len(axes_dataframe.columns) - 1)]
        # Clear any text
        cell.get_text().set_visible(False)

        # Get the bounding box of the cell in display coordinates
        bbox = cell.get_window_extent(table_ax.figure.canvas.get_renderer())

        # Convert the bounding box to figure coordinates
        bbox = table_ax.transData.inverted().transform(bbox)

        # Draw the line style over the cell
        line = Line2D(
            [
                bbox[0][0] + (bbox[1][0] - bbox[0][0]) * 0.1,
                bbox[0][0] + (bbox[1][0] - bbox[0][0]) * 0.9,
            ],
            [(bbox[0][1] + bbox[1][1]) / 2] * 2,
            color=color,
            linestyle=line_style,
            linewidth=2,
            transform=table_ax.transAxes,
        )
        table_ax.add_line(line)

    # Adjust the layout
    # fig.tight_layout()

    return fig, [*axs, table_ax]


def create_axes_parameters_tables_sequential(
    fig, axs: list, tables_list: list, table_height=0.2, spacing=0.05
) -> tuple:
    """
    Inserts tables between vertical subplots in an existing figure.
    Adjusts the subplot positions to create space for the tables.

    Parameters:
    - fig: The figure object containing the subplots.
    - axs: A list of axes objects corresponding to the subplots.
    - tables_list: A list of lists containing the data to display in the tables.
                   Each sub-list corresponds to one table.
    - table_height: The height of the tables relative to the figure (default is 0.2).
    - spacing: Space between the subplots and the tables (default is 0.05).

    Returns:
    - None
    """

    # Loop over each axis and table
    for i in range(len(axs)):
        # Convert the current table's list into a DataFrame
        df = pd.DataFrame(tables_list[i][1:], columns=tables_list[i][0])

        # Calculate the amount of space required for the table plus spacing
        total_shift = table_height + spacing

        # Shift all axes below the current one downwards
        for j in range(i + 1, len(axs)):
            pos = axs[j].get_position()
            axs[j].set_position([pos.x0, pos.y0 - total_shift, pos.width, pos.height])

        # Determine the position to insert the table
        upper_bbox = axs[i].get_position()

        # Set the table's position just below the current axis
        table_bottom = upper_bbox.y0 - total_shift + spacing

        # Add the table in the calculated position
        table_ax = fig.add_axes(
            [upper_bbox.x0, table_bottom, upper_bbox.width, table_height]
        )
        table_ax.axis("off")  # Turn off the axis for the table
        table_ax.table(
            cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
        )

    return fig, [*axs, table_ax]
