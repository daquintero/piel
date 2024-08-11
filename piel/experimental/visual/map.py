from .dc import plot_dc_sweep, plot_dc_sweeps
from .frequency import plot_s_parameter_real_and_imaginary


"""
This mapping creates an automatic relationships between the corresponding measurement data and the list of plots
that should be generated for it.
"""
measurement_data_to_plot_map = {
    "PropagationDelayMeasurementData": [plot_dc_sweep],
    "VNASParameterMeasurementData": [plot_s_parameter_real_and_imaginary],
}

"""
This mapping creates an automatic relationship between the data collection and the plotting required.
"""
measurement_data_collection_to_plot_map = {
    "PropagationDelayMeasurementDataCollection": [plot_dc_sweeps],
    "VNASParameterMeasurementDataCollection": [plot_s_parameter_real_and_imaginary],
}

measurement_data_collection_to_plot_prefix_map = {
    "PropagationDelayMeasurementDataCollection": ["plot_dc_sweeps"],
    "VNASParameterMeasurementDataCollection": ["plot_s_parameter_real_and_imaginary"],
}
