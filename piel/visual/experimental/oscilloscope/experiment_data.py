import matplotlib.pyplot as plt
from piel.visual import save
from piel.types.experimental import ExperimentData
from typing import Optional


def plot_signal_propagation_measurements(
    experiment_data: ExperimentData,
    x_parameter: str = "",
    measurement_name: str = "",
    measurement_section: Optional[list[str]] = None,
    xlabel=r"ScalarSource Frequency $GHz$",
    ylabel=r"Propagation Delay $ns$",
    yscale_factor=1e9,
    *args,
    **kwargs,
):
    import pandas as pd

    x_parameter_data = pd.DataFrame(experiment_data.experiment.parameters_list)[
        x_parameter
    ]

    signal_propagation_sweep_data = experiment_data.data.collection

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if measurement_section is None:
        measurement_section = ["value", "mean", "min", "max"]

    for measurement_section_i in measurement_section:
        x_data = list()
        y_data = list()
        i = 0
        for signal_propagation_measurement_data_i in signal_propagation_sweep_data:
            # Go through each of the files measurements to extract the relevant files
            x_data.append(x_parameter_data[i])
            y_data.append(
                getattr(
                    signal_propagation_measurement_data_i.measurements[
                        measurement_name
                    ],
                    measurement_section_i,
                )
                * yscale_factor
            )
            i += 1

        ax.plot(x_data, y_data, "o", label=measurement_section_i)

    ax.legend()
    ax.set_title("Transient Propagation Delay Characterization \n RF PCB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, ax
