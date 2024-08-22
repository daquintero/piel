from .propagation import (
    PropagationDelayMeasurementDataCollection,
    PropagationDelayMeasurementData,
)
from .frequency import (
    FrequencyMeasurementDataTypes,
    FrequencyMeasurementDataCollectionTypes,
)
from .dc import DCMeasurementDataTypes, DCMeasurementDataCollection
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
    DCMeasurementDataCollection
    | FrequencyMeasurementDataCollectionTypes
    | PropagationDelayMeasurementDataCollection
    | OscilloscopeMeasurementDataCollection
)
