from typing import Literal
from ..core import PielBaseModel, ArrayTypes


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
