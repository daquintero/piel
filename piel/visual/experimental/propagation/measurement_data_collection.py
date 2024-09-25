import piel.types
from piel.types import Unit
from piel.visual.plot.core import (
    save,
)
from piel.visual.plot.position import (
    create_plot_containers,
)
from piel.visual.plot.table import (
    create_axes_parameters_tables_separate,
)
from piel.visual.plot.basic import plot_simple
from piel.visual.style import secondary_color_palette
from piel.types.experimental import PropagationDelayMeasurementDataCollection
from piel.analysis.signals.time_data import (
    extract_rising_edges,
    resize_data_time_signal_units,
)
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


def plot_propagation_signals_time(
    data_collection: PropagationDelayMeasurementDataCollection,
    parameters_list: list = None,
    measurement_section: Optional[list[str]] = None,
    xlabel: str | Unit = None,
    ylabel: str | Unit = None,
    figure_title: str = None,
    create_parameters_tables: bool = True,
    axes_subtitle_list: list[str] = None,
    label_per_axes: bool = False,
    label_style: Literal["label_per_axes", "label_per_figure"] = "label_per_figure",
    dut_plot_kwargs: Optional[dict] = None,
    reference_plot_kwargs: Optional[dict] = None,
    figure_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    rising_edges_kwargs: Optional[dict] = None,
    *args,
    **kwargs,
):
    """
    Generates a series of plots representing the propagation signals over time, where each subplot corresponds
    to a measurement in the given data collection.

    Parameters:
    ----------
    data_collection : PropagationDelayMeasurementDataCollection
        A collection of data related to propagation delay measurements, containing signal waveforms.

    parameters_list : list, optional
        A list of parameters to be used as labels for the subplots. Defaults to the length of data_collection if None.

    measurement_section : list[str], optional
        List of sections of the measurement for further categorization.

    xlabel : str or piel.types.Unit, optional
        The label for the x-axis. If a `piel.types.Unit` object is passed, data correction is applied based on the unit.

    ylabel : str or piel.types.Unit, optional
        The label for the y-axis. If a `piel.types.Unit` object is passed, data correction is applied based on the unit.

    figure_title : str, optional
        The title of the figure. Defaults to the name of the data collection.

    create_parameters_tables : bool, optional
        If True, creates tables of parameters for each axis. Defaults to True.

    axes_subtitle_list : list[str], optional
        A list of subtitles for each axis.

    label_per_axes : bool, optional
        If True, the x and y labels will be set individually for each axis. Defaults to False.

    reference_plot_kwargs : dict, optional
        Customization options for plotting the reference signal (e.g., line style, label). Defaults to a solid line labeled "REF".

    dut_plot_kwargs : dict, optional
        Customization options for plotting the DUT signal (e.g., line style, label). Defaults to a solid line labeled "DUT".

    figure_kwargs : dict, optional
        Customization options for figure definition.

    *args, **kwargs :
        Additional arguments for plot customization, figure saving, or debugging.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.

    axs : list of matplotlib.axes.Axes
        List of axes corresponding to the subplots.

    Notes:
    -----
    - The function handles missing waveforms gracefully, skipping any missing data.
    - If units are passed for `xlabel` or `ylabel`, a correction factor is applied to adjust the plotted data.
    - Parameter tables can be created for each subplot based on the `parameters_list`.
    """
    if parameters_list is None:
        parameters_list = range(len(data_collection.collection))

    if reference_plot_kwargs is None:
        reference_plot_kwargs = {
            "linestyle": "-",
            "label": "REF",
        }

    if dut_plot_kwargs is None:
        dut_plot_kwargs = {
            "linestyle": "-",
            "label": "DUT",
        }

    if figure_kwargs is None:
        figure_kwargs = {}

    if legend_kwargs == {}:
        legend_kwargs = {"loc": "center right"}

    if isinstance(xlabel, str):
        xlabel_str = xlabel
        x_unit = piel.types.s
    elif isinstance(xlabel, Unit):
        xlabel_str = xlabel.label
        x_unit = xlabel
    else:
        xlabel_str = r"Time $s$"
        x_unit = piel.types.s

    if isinstance(ylabel, str):
        ylabel_str = ylabel
        y_unit = piel.types.V
    elif isinstance(ylabel, Unit):
        ylabel_str = ylabel.label
        y_unit = ylabel
    else:
        ylabel_str = r"Voltage $V$"
        y_unit = piel.types.V

    if figure_title is None:
        figure_title = data_collection.name

    # TODO Implement validation that it's a time-propagation delay measurement
    fig, axs = create_plot_containers(
        data_collection.collection,
        sharex=True,
        **figure_kwargs,
    )

    smallest_range = None
    smallest_start = None
    smallest_end = None

    parameter_tables_list = list()

    i = 0
    for signal_propagation_measurement_data_i in data_collection.collection:
        if signal_propagation_measurement_data_i.reference_waveform is None:
            pass
        else:
            reference_waveform = (
                signal_propagation_measurement_data_i.reference_waveform
            )
            dut_waveform = signal_propagation_measurement_data_i.dut_waveform

            reference_waveform = resize_data_time_signal_units(
                reference_waveform,
                time_unit=x_unit,
                data_unit=y_unit,
            )

            dut_waveform = resize_data_time_signal_units(
                dut_waveform,
                time_unit=x_unit,
                data_unit=y_unit,
            )

            start_time = reference_waveform.time_s[0]
            end_time = reference_waveform.time_s[-1]
            current_range = end_time - start_time

            # Update the smallest range if the current one is smaller
            if smallest_range is None or current_range < smallest_range:
                smallest_range = current_range
                smallest_start = start_time
                smallest_end = end_time

            ax = axs[i]

            # ax.set_title(parameters_list[i])
            parameter_tables_list.append(parameters_list[i])

            plot_simple(
                reference_waveform.time_s,
                reference_waveform.data,
                fig=fig,
                axs=[ax],
                label=reference_plot_kwargs["label"],
                plot_kwargs=reference_plot_kwargs,
                legend_kwargs=legend_kwargs,
                figure_kwargs=figure_kwargs,
            )
            plot_simple(
                dut_waveform.time_s,
                dut_waveform.data,
                fig=fig,
                axs=[ax],
                label=dut_plot_kwargs["label"],
                plot_kwargs=dut_plot_kwargs,
                legend_kwargs=legend_kwargs,
                figure_kwargs=figure_kwargs,
            )

            if rising_edges_kwargs is not None:
                if rising_edges_kwargs == {}:
                    rising_edges_kwargs = {
                        "linestyle": "-",
                    }

                # We want to plot these in the figure with a slightly darker color than the main color cycler.
                reference_rising_edges = extract_rising_edges(reference_waveform)
                dut_rising_edges = extract_rising_edges(dut_waveform)

                for rising_edge in reference_rising_edges:
                    plot_simple(
                        rising_edge.time_s,
                        rising_edge.data,
                        fig=fig,
                        axs=[ax],
                        plot_kwargs={
                            "color": secondary_color_palette[0],
                            **rising_edges_kwargs,
                        },
                        legend_kwargs=legend_kwargs,
                        figure_kwargs=figure_kwargs,
                    )

                for rising_edge in dut_rising_edges:
                    plot_simple(
                        rising_edge.time_s,
                        rising_edge.data,
                        fig=fig,
                        axs=[ax],
                        plot_kwargs={
                            "color": secondary_color_palette[1],
                            **rising_edges_kwargs,
                        },
                        legend_kwargs=legend_kwargs,
                        figure_kwargs=figure_kwargs,
                    )

            if label_per_axes == "label_per_axes":
                ax.set_ylabel(ylabel_str)

                if i == (len(data_collection.collection)):
                    ax.set_xlabel(xlabel_str)

        i += 1

    # After processing all waveforms, set the x-limits if a valid range was found
    if smallest_start is not None and smallest_end is not None:
        axs[0].set_xlim([smallest_start, smallest_end])
    else:
        # Optional: Handle cases where no valid waveforms are present
        print("No valid reference waveforms found to set x-limits.")

    if create_parameters_tables:
        if parameters_list is not None:
            if len(parameters_list) == len(data_collection.collection):
                # Create the labels accordingly
                try:
                    create_axes_parameters_tables_separate(
                        fig=fig, axs=axs, parameter_tables_list=parameter_tables_list
                    )
                except Exception as e:
                    if "debug" in kwargs and kwargs.get("debug", False):
                        raise e

    if axes_subtitle_list is not None:
        i = 0
        for axes_i in axes_subtitle_list:
            axs[i].set_title(axes_i, loc="left")
            i += 1

    fig.suptitle(figure_title)

    if label_style == "label_per_figure":
        fig.supxlabel(xlabel_str)
        fig.supylabel(ylabel_str)

    save(fig, **kwargs)

    return fig, axs
