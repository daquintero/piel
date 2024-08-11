"""
By default, we want all of these plots to work based on an input of a `ExperimentData`.
This is because, they contain the necessary metadata and data to generate the plots from the selected operating points.
However, we also want the flexibility of creating plots from a given set of operating conditions within the larger set.
As such, it is also necessary for this functionality to be included, and propagated into each of these functions.
Only then, it will be much easier to generate the plots from the relevant operating points within the metadata.

However, in terms of ease of API it would still make sense to provide a generic collection of measurements,
and the corresponding metadata only directly to the plots.
"""

from .dc import plot_dc_sweep, plot_dc_sweeps
from .propagation import (
    plot_signal_propagation_measurements,
    plot_signal_propagation_signals,
)
from .frequency import (
    plot_s_parameter_measurements_to_step_responses,
    plot_s_parameter_real_and_imaginary,
)
from .auto import (
    auto_plot_from_measurement_data_collection,
    auto_plot_from_measurement_data,
    auto_plot_from_experiment_data,
)
