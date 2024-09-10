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
from .oscilloscope import (
    OscilloscopeMeasurement,
    OscilloscopeMeasurementConfiguration,
    OscilloscopeMeasurementCollection,
)

# Configuration
TimeMeasurementConfigurationTypes = (
    PropagationDelayMeasurementConfiguration | OscilloscopeMeasurementConfiguration
)
FrequencyMeasurementConfigurationTypes = (
    VNASParameterMeasurementConfiguration | VNAPowerSweepMeasurementConfiguration
)
MeasurementConfigurationTypes = (
    TimeMeasurementConfigurationTypes | FrequencyMeasurementConfigurationTypes
)

# Measurements
FrequencyMeasurementTypes = VNASParameterMeasurement | VNAPowerSweepMeasurement
TimeMeasurementTypes = OscilloscopeMeasurement | PropagationDelayMeasurement
MeasurementTypes = TimeMeasurementTypes | FrequencyMeasurementTypes

# Measurement Collections
MeasurementCollectionTypes = (
    PropagationDelayMeasurementCollection
    | OscilloscopeMeasurementCollection
    | VNASParameterMeasurementCollection
    | VNAPowerSweepMeasurementCollection
)
