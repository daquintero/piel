from piel.types.experimental import (
    VNASParameterMeasurement,
    VNASParameterMeasurementCollection,
)
from piel.types.experimental import (
    OscilloscopeMeasurement,
    OscilloscopeMeasurementCollection,
)
from piel.types.experimental import (
    PropagationDelayMeasurement,
    PropagationDelayMeasurementCollection,
)


from piel.types.experimental import (
    VNASParameterMeasurementData,
    VNASParameterMeasurementDataCollection,
)
from piel.types.experimental import (
    OscilloscopeMeasurementData,
    OscilloscopeMeasurementDataCollection,
)
from piel.types.experimental import (
    PropagationDelayMeasurementData,
    PropagationDelayMeasurementDataCollection,
)


from piel.experimental.measurements.propagation import compose_propagation_delay_measurement
from piel.experimental.measurements.oscilloscope import compose_oscilloscope_measurement
from piel.experimental.measurements.frequency import compose_vna_s_parameter_measurement

from piel.experimental.measurements.data.frequency import extract_s_parameter_data_from_vna_measurement
from piel.experimental.measurements.data.oscilloscope import extract_oscilloscope_data_from_measurement
from piel.experimental.measurements.data.propagation import extract_propagation_delay_data_from_measurement


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
