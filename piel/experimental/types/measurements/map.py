from .frequency import (
    VNAPowerSweepMeasurementConfiguration,
    VNASParameterMeasurementConfiguration,
    VNASParameterMeasurement,
)

from .propagation import (
    PropagationDelayMeasurementConfiguration,
    PropagationDelayMeasurement,
)

from .data.propagation import PropagationDelayMeasurementData

# Note that the configuration and measurement should have the same fields without _prefix
configuration_to_measurement_map = {
    PropagationDelayMeasurementConfiguration.__name__: PropagationDelayMeasurement,
    VNASParameterMeasurementConfiguration.__name__: VNASParameterMeasurement,
    VNAPowerSweepMeasurementConfiguration.__name__: VNASParameterMeasurement,
}

measurement_to_data_map = {
    PropagationDelayMeasurement.__name__: PropagationDelayMeasurementData
}
