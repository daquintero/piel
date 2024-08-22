from ..types.measurements.frequency import (
    VNASParameterMeasurement,
    VNASParameterMeasurementCollection,
)
from ..types.measurements.oscilloscope import (
    OscilloscopeMeasurement,
    OscilloscopeMeasurementCollection,
)
from ..types.measurements.propagation import (
    PropagationDelayMeasurement,
    PropagationDelayMeasurementCollection,
)


from ..types.measurements.data.frequency import (
    VNASParameterMeasurementData,
    VNASParameterMeasurementDataCollection,
)
from ..types.measurements.data.oscilloscope import (
    OscilloscopeMeasurementData,
    OscilloscopeMeasurementDataCollection,
)
from ..types.measurements.data.propagation import (
    PropagationDelayMeasurementData,
    PropagationDelayMeasurementDataCollection,
)


from .propagation import compose_propagation_delay_measurement
from .oscilloscope import compose_oscilloscope_measurement
from .frequency import compose_vna_s_parameter_measurement

from .data.frequency import extract_s_parameter_data_from_vna_measurement
from .data.oscilloscope import extract_oscilloscope_data_from_measurement
from .data.propagation import extract_propagation_delay_data_from_measurement


# Note that the configuration and measurement should have the same fields without _prefix
configuration_to_measurement_map = {
    "OscilloscopeMeasurementConfiguration": OscilloscopeMeasurement,
    "PropagationDelayMeasurementConfiguration": PropagationDelayMeasurement,
    "VNASParameterMeasurementConfiguration": VNASParameterMeasurement,
    "VNAPowerSweepMeasurementConfiguration": VNASParameterMeasurement,
}

measurement_composition_method_mapping = {
    "VNASParameterMeasurementConfiguration": compose_vna_s_parameter_measurement,
    "PropagationDelayMeasurementConfiguration": compose_propagation_delay_measurement,
    "OscilloscopeMeasurementConfiguration": compose_oscilloscope_measurement,
}

measurement_to_data_map = {
    "OscilloscopeMeasurement": OscilloscopeMeasurementData,
    "PropagationDelayMeasurement": PropagationDelayMeasurementData,
    "VNASParameterMeasurement": VNASParameterMeasurementData,
}

measurement_to_data_method_map = {
    "OscilloscopeMeasurement": extract_oscilloscope_data_from_measurement,
    "PropagationDelayMeasurement": extract_propagation_delay_data_from_measurement,
    "VNASParameterMeasurement": extract_s_parameter_data_from_vna_measurement,
}

measurement_to_collection_map = {
    "OscilloscopeMeasurement": OscilloscopeMeasurementCollection,
    "PropagationDelayMeasurement": PropagationDelayMeasurementCollection,
    "VNASParameterMeasurement": VNASParameterMeasurementCollection,
}

measurement_collection_to_data_map = {
    "OscilloscopeMeasurementCollection": OscilloscopeMeasurementDataCollection,
    "PropagationDelayMeasurementCollection": PropagationDelayMeasurementDataCollection,
    "VNASParameterMeasurementCollection": VNASParameterMeasurementDataCollection,
}

measurement_data_to_measurement_collection_data_map = {
    "OscilloscopeMeasurementData": OscilloscopeMeasurementDataCollection,
    "PropagationDelayMeasurementData": PropagationDelayMeasurementDataCollection,
    "VNASParameterMeasurementData": VNASParameterMeasurementDataCollection,
}
