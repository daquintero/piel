def create_axes_parameters_table(ax, parameters_list):
    """
    This function takes in the parameter_list and a figure, axes pair to return a figure and axes with an attached
    parameter table and relevant colors. The figure must already have the axes plotted and ready to extract the
    relevant colors and parameters from it accordingly, these need to be extracted from the corresponding lines list.
    """
    # Step 4: Add a subplot or axes for the DataFrame beside the plots
    import pandas as pd
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    lines_list = ax.get_lines()
    colors = [line.get_color() for line in lines_list]

    # Step 3: Combine the parameters_list with extracted labels and colors
    plot_line_data = []
    for parameters_i, color in zip(parameters_list, colors):
        plot_line_entry = (
            parameters_i.copy()
        )  # Copy the original dictionary to avoid modifying it
        plot_line_entry["Color"] = color
        plot_line_data.append(plot_line_entry)

    axes_dataframe = pd.DataFrame(plot_line_data)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="30%", pad=0.05
    )  # Create new axes on the right side

    # Remove the new axes border and ticks
    cax.axis("off")

    # Create a table within the new axes
    cax.table(
        cellText=axes_dataframe.values,
        colLabels=axes_dataframe.columns,
        loc="center",
        cellLoc="center",
    )
