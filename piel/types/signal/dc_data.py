from ..core import ArrayTypes
from ..connectivity.abstract import Instance
from ..units import Unit, ratio


class SignalTraceDC(Instance):
    """
    Represents a DC signal with all relevant components as defined by Ohm's law but specified through a collection of data defined by a `OperatingPointContainer`

    The values are the values of the signal. The name is the name of the signal. Can be both an operating point
    or a sweep collection of data.

    A DC signal might have a current and a voltage attached to it, so it would be a collection of data.
    Current and voltage are both physical representations of electrical quantities in this case in the context of DC operation.
    These operating points can reference an array of data points that are collected from a DC sweep.
    """

    unit: Unit = ratio
    """
    Intended for DC electrical types such as voltage, current, power and resistance.
    """

    values: ArrayTypes = []
    """
    The values of the operating points in an array format.
    """


class SignalDC(Instance):
    """
    This is used to define a collection of `SignalInstances` which compose a DC signal. For example,
    the voltage and current of the same signal would be `SignalInstance`s but the total signal is the collection of
    these data references.

    These DC signals may refer to a single trace, with a collection of voltage, current, resistance or power data points.
    These may still be referencing the original node.
    """

    trace_list: list[SignalTraceDC] = []


class SignalDCCollection(Instance):
    inputs: list[SignalDC] | list[SignalDC] = []
    """
    The input DC signals.
    """

    outputs: list[SignalDC] | list[SignalDC] = []
    """
    The output DC signals.
    """
