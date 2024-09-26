from piel.types.experimental import DCSweepMeasurementDataCollection
from piel.visual import create_plot_containers, save


def plot_two_port_dc_sweep(
    dc_sweep_collection: DCSweepMeasurementDataCollection,
    fig=None,
    axs=None,
    title: str = "",
    label_list: list = None,
    **kwargs,
) -> tuple:
    """
    This will always plot on the first axes. Provide subset if desired.
    """
    if (fig is None) and (axs is None):
        fig, axs = create_plot_containers(
            container_list=dc_sweep_collection.collection, axes_structure="overlay"
        )

    i = 0
    for dc_sweep_i in dc_sweep_collection.collection:
        if label_list is not None:
            # TODO improve this.
            axs[0].plot(
                dc_sweep_i.inputs[0].trace_list[0].values,
                dc_sweep_i.outputs[0].trace_list[0].values,
                label=r"$V_{dd}$" + f" = {label_list[i]}",
            )
        else:
            axs[0].plot(
                dc_sweep_i.inputs[0].trace_list[0].values,
                dc_sweep_i.outputs[0].trace_list[0].values,
            )

        i += 1

    axs[0].legend(loc="lower right")
    axs[0].set_xlabel(r"$V_{in}$ $V$")
    axs[0].set_ylabel("\n" + r"$V_{out}$ $V$")
    axs[0].set_title(title)

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs
