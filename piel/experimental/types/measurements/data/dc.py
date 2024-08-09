from .core import MeasurementData
from .....types import SignalDC


class SourcemeterSweepMeasurementData(MeasurementData):
    signal: SignalDC = None


class MultimeterSweepVoltageMeasurementData(MeasurementData):
    signal: SignalDC = None


class DCSweepMeasurementData(MeasurementData):
    inputs: list[SourcemeterSweepMeasurementData] = None
    """
    The input DC signals as sourced by a sourcemeter.
    """

    outputs: list[MultimeterSweepVoltageMeasurementData] = None
    """
    The output DC signals from a multimeter for example.
    """


DCSweepMeasurementDataCollection = list[DCSweepMeasurementData]

DCMeasurementDataTypes = (
    DCSweepMeasurementData
    | MultimeterSweepVoltageMeasurementData
    | SourcemeterSweepMeasurementData
)

DCMeasurementDataCollection = (
    list[DCMeasurementDataTypes] | DCSweepMeasurementDataCollection
)
