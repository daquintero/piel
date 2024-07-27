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
