from .propagation import (
    PropagationDelayMeasurementDataCollection,
    PropagationDelayMeasurementData,
)
from .frequency import (
    FrequencyMeasurementDataTypes,
    FrequencyMeasurementDataCollectionTypes,
)
from .dc import DCMeasurementDataTypes, DCMeasurementDataCollectionTypes
from .oscilloscope import (
    OscilloscopeMeasurementData,
    OscilloscopeMeasurementDataCollection,
)


MeasurementDataTypes = (
    DCMeasurementDataTypes
    | FrequencyMeasurementDataTypes
    | PropagationDelayMeasurementData
    | OscilloscopeMeasurementData
)

# Measurement Collections
MeasurementDataCollectionTypes = (
    DCMeasurementDataCollectionTypes
    | FrequencyMeasurementDataCollectionTypes
    | PropagationDelayMeasurementDataCollection
    | OscilloscopeMeasurementDataCollection
)
