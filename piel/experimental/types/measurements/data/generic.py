from .propagation import (
    PropagationDelayMeasurementDataCollection,
    PropagationDelayMeasurementData,
)
from .frequency import FrequencyMeasurementDataTypes, FrequencyMeasurementDataCollection
from .dc import DCMeasurementDataTypes, DCMeasurementDataCollection

MeasurementDataTypes = (
    DCMeasurementDataTypes
    | FrequencyMeasurementDataTypes
    | PropagationDelayMeasurementData
)

# Measurement Collections
MeasurementDataCollectionTypes = (
    DCMeasurementDataCollection
    | FrequencyMeasurementDataCollection
    | PropagationDelayMeasurementDataCollection
)
