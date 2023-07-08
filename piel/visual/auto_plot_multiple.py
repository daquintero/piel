import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "plot_simple_multi_row",
    "plot_multi_row",
]


def plot_simple_multi_row(
    data: pd.DataFrame,
    x_axis_column_name: str = "t",
    row_list: list | None = None,
    y_axis_title_list: list | None = None,
    x_axis_title: str | None = None,
):
    """
    Plot multiple rows of data on the same plot. Each row is a different line. Each row is a different y axis. The x
    axis is the same for all rows. The y axis title is the same for all rows.

    Args:
        data (pd.DataFrame): Data to plot.
        x_axis_column_name (str, optional): Column name of the x axis. Defaults to "t".
        row_list (list, optional): List of column names to plot. Defaults to None.
        y_axis_title_list (list, optional): List of y axis titles. Defaults to None.
        x_axis_title (str, optional): Title of the x axis. Defaults to None.

    Returns:
        plt: Matplotlib plot.
    """
    x = data[x_axis_column_name]
    y_array = []

    if y_axis_title_list is None:
        y_axis_title_list = row_list

    row_amount = len(row_list)
    for row_name in row_list:
        y_array.append(data[row_name])

    fig, axes = plt.subplots(row_amount, 1, sharex=True)

    for i in range(len(row_list)):
        axes[i].plot(x, y_array[i])
        axes[i].grid(True)
        axes[i].set(ylabel=y_axis_title_list[i])

    # TODO Xaxis title
    # TODO align all ytitles

    return plt


def plot_multi_row(
    data: pd.DataFrame,
):
    pass
    # from bokeh.plotting import figure, show
    # from bokeh.layouts import column
    #
    # p = figure(
    #     width=800,
    #     height=300,
    #     title="",
    #     tools="",
    #     toolbar_location=None,
    #     match_aspect=True,
    #     y_range=[0, 1],
    # )
    # p2 = figure(width=800, height=300, x_range=p.x_range)
    # p3 = figure(width=800, height=300, x_range=p.x_range)
    #
    # p.line(
    #     mzi2x2_simple_simulation_data.t / 1000,
    #     mzi2x2_simple_simulation_data.output_amplitude_array_0_abs,
    # )
    # # color="navy", alpha=0.4, line_width=4)
    #
    # p.line(
    #     mzi2x2_simple_simulation_data.t / 1000,
    #     mzi2x2_simple_simulation_data.output_amplitude_array_1_abs,
    # )
    # # color="navy", alpha=0.4, line_width=4)
    #
    # p2.line(
    #     mzi2x2_simple_simulation_data.t / 1000,
    #     mzi2x2_simple_simulation_data.output_amplitude_array_0_phase_deg,
    # )
    # # color="navy", alpha=0.4, line_width=4)
    #
    # p2.line(
    #     mzi2x2_simple_simulation_data.t / 1000,
    #     mzi2x2_simple_simulation_data.output_amplitude_array_1_phase_deg,
    # )
    #
    # p3.line(
    #     mzi2x2_simple_simulation_data.t / 1000,
    #     mzi2x2_simple_simulation_data.phase,
    # )
    # # color="navy", alpha=0.4, line_width=4)
    #
    # # color="navy", alpha=0.4, line_width=4)
    #
    # # show(p)
    # # layout = gridplot([[p], [p2]])
    # return show(column(p, p2, p3))
