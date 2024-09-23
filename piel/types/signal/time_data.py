from typing import Optional, Literal
from ..core import PielBaseModel, ArrayTypes, NumericalTypes
from piel.types.connectivity.timing import TimeMetrics


class SignalMetricsData(TimeMetrics):
    """
    Standard definition for a signal measurement. It includes the value, mean, min, max, standard deviation and count.
    """

    count: Optional[NumericalTypes] | Optional[str]


SignalMetricsMeasurementCollection = dict[str, SignalMetricsData]
"""
Collection of SignalMeasurements that can be used to analyse a set of signals together.
"""


class DataTimeSignalData(PielBaseModel):
    """
    Standard definition for a relationship between a relevant files signal and a time reference array.
    Sources could be both measurement and simulation.
    """

    time_s: ArrayTypes = []
    data: ArrayTypes = []
    data_name: str = ""


MultiDataTimeSignal = list[DataTimeSignalData, ...]
"""
Collection of DataTimeSignals that can be used to analyse a set of signals together in a particular files flow.
"""

EdgeTransitionAnalysisTypes = Literal["mean", "peak_to_peak"]
