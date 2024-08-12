"""
By default, we want all of these plots to work based on an input of a `ExperimentData`.
This is because, they contain the necessary metadata and data to generate the plots from the selected operating points.
However, we also want the flexibility of creating plots from a given set of operating conditions within the larger set.
As such, it is also necessary for this functionality to be included, and propagated into each of these functions.
Only then, it will be much easier to generate the plots from the relevant operating points within the metadata.

However, in terms of ease of API it would still make sense to provide a generic collection of measurements,
and the corresponding metadata only directly to the plots.

It would make sense to split this functionality into subsections or subfunctions for the goal of logically reusing code,
whilst still having extensible flexibility. As such, it might, for example, be desired to provide a set of functionality for
`ExperimentDataCollection` that operates onto each `ExperimentData`. Hence, it would make sense to have a set of plots
according to the type of data input provided. As such, maybe it makes sense to have submodules of the corresponding
plots rather than a large subset, with the functionality being submodule dependent.

By having the plotting be implemented at multiple levels of data, then full parametrization can be achieved.


Note that all the functions inside the ``experiment_data`` just take the raw data collection and does not perform extraction by default.
"""

from . import dc
from . import frequency
from . import propagation

from .map import (
    measurement_data_to_plot_map,
    measurement_data_collection_to_plot_map,
    measurement_data_collection_to_plot_suffix_map,
    auto_function_name_list_from_module,
    auto_function_list_from_module,
)
from .auto import (
    auto_plot_from_measurement_data_collection,
    auto_plot_from_measurement_data,
    auto_plot_from_experiment_data,
)

# TODO depreciate this
# from piel.experimental.visual.dc.dc import plot_dc_sweep, plot_dc_sweeps
# from piel.experimental.visual.propagation.propagation import (
#     plot_signal_propagation_measurements,
#     plot_propagation_signals_time,
# )
# from piel.experimental.visual.frequency.frequency import (
#     plot_s_parameter_measurements_to_step_responses,
#     plot_s_parameter_real_and_imaginary,
#     plot_s_parameter_per_component,
# )
