from typing import Optional
from ..core import PielBaseModel, ArrayTypes, NumericalTypes


class SignalMetricsData(PielBaseModel):
    """
    Standard definition for a signal measurement. It includes the value, mean, min, max, standard deviation and count.
    """

    value: NumericalTypes
    mean: Optional[NumericalTypes]
    min: Optional[NumericalTypes]
    max: Optional[NumericalTypes]
    standard_deviation: Optional[NumericalTypes]
    count: Optional[NumericalTypes] | Optional[str]


SignalMetricsMeasurementCollection = dict[str, SignalMetricsData]
"""
Collection of SignalMeasurements that can be used to analyse a set of signals together.
"""


class DataTimeSignalData(PielBaseModel):
    """
    Standard definition for a relationship between a relevant files signal and a time reference array.
    Sources could be both experimental and simulation.
    """

    time_s: ArrayTypes
    data: ArrayTypes
    data_name: Optional[str]


MultiDataTimeSignal = list[DataTimeSignalData, ...]
"""
Collection of DataTimeSignals that can be used to analyse a set of signals together in a particular files flow.
"""


class SignalPropagationData(PielBaseModel):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.

    The collection includes the device waveform, the measurement files and the reference waveform as per a propagation delay measurement.

    Attributes:
        measurements (Optional[SignalMetricsMeasurementCollection]): The collection of signal measurements.
        device_waveform (Optional[DataTimeSignalData]): The device waveform.
        reference_waveform (Optional[DataTimeSignalData]): The reference waveform.
        source_frequency_GHz (Optional[float]): The source frequency in
            gigahertz (GHz).
    """

    measurements: Optional[SignalMetricsMeasurementCollection]
    device_waveform: Optional[DataTimeSignalData]
    reference_waveform: Optional[DataTimeSignalData]

    source_frequency_GHz: Optional[float] = None


class SignalPropagationSweepData(PielBaseModel):
    """
    This class is used to define a collection of PropagationDelayData that are part of a sweep of a parameter
    as defined within each PropagationDelayData.

    Attributes:
        sweep_parameter_name (str): The name of the parameter that is being swept. Must exist within the PropagationDelayData files definition.
        data (list[SignalPropagationData]): The collection of PropagationDelay
    """

    sweep_parameter_name: str
    """
    The name of the parameter that is being swept. Must exist within the PropagationDelayFileCollection files definition.
    """
    data: list[SignalPropagationData]
