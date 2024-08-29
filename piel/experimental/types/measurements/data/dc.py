from .core import MeasurementData, MeasurementDataCollection
from .....types import SignalDC


class SourcemeterSweepMeasurementData(MeasurementData):
    signal: SignalDC = None


class MultimeterSweepVoltageMeasurementData(MeasurementData):
    signal: SignalDC = None


class DCSweepMeasurementData(MeasurementData):
    type: str = "DCSweepMeasurementData"
    inputs: list[SourcemeterSweepMeasurementData] = []
    """
    The input DC signals as sourced by a sourcemeter.
    """

    outputs: list[MultimeterSweepVoltageMeasurementData] = []
    """
    The output DC signals from a multimeter for example.
    """


DCMeasurementDataTypes = (
    DCSweepMeasurementData
    | MultimeterSweepVoltageMeasurementData
    | SourcemeterSweepMeasurementData
)


class DCSweepMeasurementDataCollection(MeasurementDataCollection):
    type: str = "DCSweepMeasurementDataCollection"
    collection: list[DCSweepMeasurementData] = []


class DCMeasurementDataCollection(MeasurementDataCollection):
    type: str = "DCMeasurementDataCollection"
    collection: list[DCMeasurementDataTypes] | DCSweepMeasurementDataCollection = []


DCMeasurementDataCollectionTypes = (
    DCSweepMeasurementDataCollection | DCMeasurementDataCollection
)
