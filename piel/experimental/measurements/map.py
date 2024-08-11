from ..types.measurements.frequency import (
    VNASParameterMeasurement,
    VNASParameterMeasurementCollection,
)

from ..types.measurements.propagation import (
    PropagationDelayMeasurement,
    PropagationDelayMeasurementCollection,
)

from ..types.measurements.data.propagation import (
    PropagationDelayMeasurementData,
    PropagationDelayMeasurementDataCollection,
)
from ..types.measurements.data.frequency import (
    VNASParameterMeasurementData,
    VNASParameterMeasurementDataCollection,
)

from .propagation import compose_propagation_delay_measurement
from .frequency import compose_vna_s_parameter_measurement

from .data.frequency import extract_s_parameter_data_from_vna_measurement
from .data.propagation import extract_propagation_delay_from_measurement

# Note that the configuration and measurement should have the same fields without _prefix
configuration_to_measurement_map = {
    "PropagationDelayMeasurementConfiguration": PropagationDelayMeasurement,
    "VNASParameterMeasurementConfiguration": VNASParameterMeasurement,
    "VNAPowerSweepMeasurementConfiguration": VNASParameterMeasurement,
}

measurement_composition_method_mapping = {
    "VNASParameterMeasurementConfiguration": compose_vna_s_parameter_measurement,
    "PropagationDelayMeasurementConfiguration": compose_propagation_delay_measurement,
}

measurement_to_data_map = {
    "PropagationDelayMeasurement": PropagationDelayMeasurementData,
    "VNASParameterMeasurement": VNASParameterMeasurementData,
}

measurement_to_data_method_map = {
    "PropagationDelayMeasurement": extract_propagation_delay_from_measurement,
    "VNASParameterMeasurement": extract_s_parameter_data_from_vna_measurement,
}

measurement_to_collection_map = {
    "PropagationDelayMeasurement": PropagationDelayMeasurementCollection,
    "VNASParameterMeasurement": VNASParameterMeasurementCollection,
}

measurement_collection_to_data_map = {
    "PropagationDelayMeasurementCollection": PropagationDelayMeasurementDataCollection,
    "VNASParameterMeasurementCollection": VNASParameterMeasurementDataCollection,
}

measurement_data_to_measurement_collection_data_map = {
    "PropagationDelayMeasurementData": PropagationDelayMeasurementDataCollection,
    "VNASParameterMeasurementData": VNASParameterMeasurementDataCollection,
}
