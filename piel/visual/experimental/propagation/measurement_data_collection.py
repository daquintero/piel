from piel.types import Unit
from piel.visual import (
    save,
    create_plot_containers,
    create_axes_parameters_tables_separate,
)
from piel.types.experimental import PropagationDelayMeasurementDataCollection
from typing import Optional
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
    legend_loc: str = None,
    label_per_axes: bool = False,
    dut_plot_kwargs: Optional[dict] = None,
    reference_plot_kwargs: Optional[dict] = None,
    figure_kwargs: Optional[dict] = None,
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

    legend_loc : str, optional
        Location for the legend in each plot. Defaults to "center right".

    label_per_axes : bool, optional
        If True, the x and y labels will be set individually for each axis. Defaults to False.

    reference_plot_kwargs : dict, optional
        Customization options for plotting the reference signal (e.g., line style, label). Defaults to a solid line labeled "REF".

    dut_plot_kwargs : dict, optional
        Customization options for plotting the DUT signal (e.g., line style, label). Defaults to a solid line labeled "DUT".


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
    x_correction = 1
    y_correction = 1

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

    if xlabel is None:
        xlabel = r"Time $s$"
    elif isinstance(xlabel, str):
        pass
    elif isinstance(xlabel, Unit):
        x_correction = xlabel.base
        logger.warning(
            f"Data correction of 1/{x_correction} from unit definition {xlabel} will be applied on x-axis"
        )
        xlabel = xlabel.label

    if ylabel is None:
        ylabel = r"Voltage $V$"
    elif isinstance(ylabel, str):
        pass
    elif isinstance(ylabel, Unit):
        y_correction = ylabel.base
        logger.warning(
            f"Data correction of 1/{y_correction} from unit definition {ylabel} will be applied on y-axis."
        )
        ylabel = ylabel.label

    if figure_title is None:
        figure_title = data_collection.name

    if legend_loc is None:
        legend_loc = "center right"

    # TODO Implement validation that it's a time-propagation delay measurement
    fig, axs = create_plot_containers(
        data_collection.collection,
        sharex=True,
        **figure_kwargs,
    )

    signal_propagation_sweep_data = data_collection.collection

    # Manage missing data here
    if signal_propagation_sweep_data[1].reference_waveform is None:
        pass
    else:
        axs[0].set_xlim(
            [
                signal_propagation_sweep_data[1].reference_waveform.time_s[0]
                / x_correction,
                signal_propagation_sweep_data[1].reference_waveform.time_s[-1]
                / x_correction,
            ]
        )

    parameter_tables_list = list()
    reference_x_data = list()
    reference_y_data = list()
    dut_x_data = list()
    dut_y_data = list()

    i = 0
    for signal_propagation_measurement_data_i in data_collection.collection:
        if signal_propagation_measurement_data_i.reference_waveform is None:
            pass
        else:
            # Go through each of the files measurements to extract the relevant files
            reference_x_data = (
                signal_propagation_measurement_data_i.reference_waveform.time_s
                / x_correction
            )
            reference_y_data = (
                signal_propagation_measurement_data_i.reference_waveform.data
                / y_correction
            )
            dut_x_data = (
                signal_propagation_measurement_data_i.dut_waveform.time_s / x_correction
            )
            dut_y_data = (
                signal_propagation_measurement_data_i.dut_waveform.data / y_correction
            )

            ax = axs[i]

            # ax.set_title(parameters_list[i])
            parameter_tables_list.append(parameters_list[i])

            ax.plot(
                reference_x_data,
                reference_y_data,
                **reference_plot_kwargs,
            )
            ax.plot(dut_x_data, dut_y_data, **dut_plot_kwargs)

            if legend_loc:
                ax.legend(loc=legend_loc)

            if label_per_axes:
                ax.set_ylabel(ylabel)

                if i == (len(data_collection.collection)):
                    ax.set_xlabel(xlabel)

        i += 1

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

    if not label_per_axes:
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)

    save(fig, **kwargs)

    return fig, axs
