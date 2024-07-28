from typing import Optional
from .core import MeasurementData
from .....types import SignalMetricsMeasurementCollection, DataTimeSignalData, Instance


class PropagationDelayMeasurementData(MeasurementData):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.

    The collection includes the device waveform, the measurement files and the reference waveform as per a propagation delay measurement.

    Attributes:
        measurements (Optional[SignalMetricsMeasurementCollection]): The collection of signal measurements.
        dut_waveform (Optional[DataTimeSignalData]): The device waveform.
        reference_waveform (Optional[DataTimeSignalData]): The reference waveform.
    """

    measurements: Optional[SignalMetricsMeasurementCollection]
    dut_waveform: Optional[DataTimeSignalData]
    reference_waveform: Optional[DataTimeSignalData]


# TODO modify this for the new structure
class PropagationDelayMeasurementSweepData(Instance):
    """
    This class is used to define a collection of PropagationDelayData that are part of a sweep of a parameter
    as defined within each PropagationDelayData.

    Attributes:
        data (list[SignalPropagationData]): The collection of PropagationDelay
    """

    """
    The name of the parameter that is being swept. Must exist within the PropagationDelayMeasurement files definition.
    """
    data: list[PropagationDelayMeasurementData]
