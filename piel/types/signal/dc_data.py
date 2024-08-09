from ..core import PielBaseModel, ArrayTypes
from .core import QuantityTypesDC
from ..connectivity.abstract import Instance


class SignalInstanceMetadataDC(PielBaseModel):
    name: str = None
    """
    The name of the signal.
    """

    data_type: QuantityTypesDC = "voltage"
    """
    The type of data that the DC operating point represents.
    """


class SignalInstanceDC(SignalInstanceMetadataDC):
    """
    Represents a DC signal with all relevant components as defined by Ohm's law but specified through a collection of data defined by a `OperatingPointContainer`

    The values are the values of the signal. The name is the name of the signal. Can be both an operating point
    or a sweep collection of data.

    A DC signal might have a current and a voltage attached to it, so it would be a collection of data.
    Current and voltage are both physical representations of electrical quantities in this case in the context of DC operation.
    These operating points can reference an array of data points that are collected from a DC sweep.
    """

    values: ArrayTypes
    """
    The values of the operating points in an array format.
    """


class SignalDC(Instance):
    """
    This is used to define a collection of `SignalInstances` which compose a DC signal. For example,
    the voltage and current of the same signal would be `SignalInstance`s but the total signal is the collection of
    these data references.
    """

    signal_instances: list[SignalInstanceDC]
