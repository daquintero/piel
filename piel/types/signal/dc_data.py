from typing import Optional
from ..core import PielBaseModel, ArrayTypes


class SignalDC(PielBaseModel):
    """
    Represents a DC signal.

    The values are the values of the signal. The name is the name of the signal. Can be both an operating point
    or a sweep collection of data.
    """

    name: Optional[str]
    """
    The name of the signal.
    """

    values: ArrayTypes
    """
    The values of the signal.
    """


class DCSweepData(PielBaseModel):
    inputs: list[SignalDC]
    """
    The input DC signals.
    """

    outputs: list[SignalDC]
    """
    The output DC signals.
    """
