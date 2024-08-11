from .propagation import (
    PropagationDelayMeasurementDataCollection,
    PropagationDelayMeasurementData,
)
from .frequency import (
    FrequencyMeasurementDataTypes,
    FrequencyMeasurementDataCollectionTypes,
)
from .dc import DCMeasurementDataTypes, DCMeasurementDataCollection

MeasurementDataTypes = (
    DCMeasurementDataTypes
    | FrequencyMeasurementDataTypes
    | PropagationDelayMeasurementData
)

# Measurement Collections
MeasurementDataCollectionTypes = (
    DCMeasurementDataCollection
    | FrequencyMeasurementDataCollectionTypes
    | PropagationDelayMeasurementDataCollection
)
