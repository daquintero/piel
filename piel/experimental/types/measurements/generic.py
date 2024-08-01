from .frequency import (
    VNASParameterMeasurementConfiguration,
    VNAPowerSweepMeasurementConfiguration,
    VNASParameterMeasurement,
    VNAPowerSweepMeasurement,
    VNASParameterMeasurementCollection,
    VNAPowerSweepMeasurementCollection,
)
from .propagation import (
    PropagationDelayMeasurement,
    PropagationDelayMeasurementConfiguration,
    PropagationDelayMeasurementCollection,
)

# Configuration
FrequencyMeasurementConfigurationTypes = (
    VNASParameterMeasurementConfiguration | VNAPowerSweepMeasurementConfiguration
)
MeasurementConfigurationTypes = (
    PropagationDelayMeasurementConfiguration | FrequencyMeasurementConfigurationTypes
)

# Measurements
FrequencyMeasurementTypes = VNASParameterMeasurement | VNAPowerSweepMeasurement
MeasurementTypes = PropagationDelayMeasurement | FrequencyMeasurementTypes

# Measurement Collections
MeasurementCollectionTypes = (
    PropagationDelayMeasurementCollection
    | VNASParameterMeasurementCollection
    | VNAPowerSweepMeasurementCollection
)
