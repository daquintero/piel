from .core import MeasurementData, MeasurementDataCollection
from piel.types.signal.time_data import (
    DataTimeSignalData,
)
from piel.types.metrics import ScalarMetricCollection


class OscilloscopeMeasurementData(MeasurementData):
    """
    Standard definition for a collection of files that are part of a generic oscilloscpe measurement

    The collection includes a list of waveform files, and a measurements file.

    Attributes:
        measurements (Optional[SignalMetricsMeasurementCollection]): The collection of signal measurements.
        waveform_list (list[DataTimeSignalData]): The list of waveforms.
    """

    type: str = "OscilloscopeMeasurementData"
    measurements: ScalarMetricCollection | None = None
    waveform_list: list[DataTimeSignalData] = []


class OscilloscopeMeasurementDataCollection(MeasurementDataCollection):
    type: str = "OscilloscopeMeasurementDataCollection"
    collection: list[OscilloscopeMeasurementData] = []
