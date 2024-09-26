from .core import MeasurementDataCollection
from piel.types.signal.dc_data import SignalDCCollection, SignalDC

DCMeasurementDataTypes = SignalDCCollection | SignalDC


class DCSweepMeasurementDataCollection(MeasurementDataCollection):
    collection: list[SignalDCCollection] = []


class DCMeasurementDataCollection(MeasurementDataCollection):
    collection: list[DCMeasurementDataTypes] | DCSweepMeasurementDataCollection = []


DCMeasurementDataCollectionTypes = (
    DCSweepMeasurementDataCollection | DCMeasurementDataCollection
)
