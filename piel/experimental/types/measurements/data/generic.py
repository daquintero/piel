from .propagation import PropagationDelayMeasurementDataCollection, PropagationDelayMeasurementData
from .frequency import FrequencyMeasurementDataTypes, FrequencyMeasurementDataCollection

MeasurementDataTypes = PropagationDelayMeasurementData | FrequencyMeasurementDataTypes

# Measurement Collections
MeasurementDataCollectionTypes = (
    PropagationDelayMeasurementDataCollection |
    FrequencyMeasurementDataCollection
)
