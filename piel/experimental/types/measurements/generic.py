from .frequency import (
    VNASParameterMeasurementConfiguration,
    VNAPowerSweepMeasurementConfiguration,
)
from .propagation import PropagationDelayMeasurement

FrequencyMeasurementConfigurationTypes = (
    VNASParameterMeasurementConfiguration | VNAPowerSweepMeasurementConfiguration
)
MeasurementConfigurationTypes = FrequencyMeasurementConfigurationTypes
MeasurementTypes = PropagationDelayMeasurement
