from .core import MeasurementData, MeasurementDataCollection
from piel.types.signal.time_data import (
    DataTimeSignalData,
)
from piel.types.metrics import ScalarMetricCollection


class PropagationDelayMeasurementData(MeasurementData):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.

    The collection includes the device waveform, the measurement files and the reference waveform as per a propagation delay measurement.

    Attributes:
        measurements (Optional[SignalMetricsMeasurementCollection]): The collection of signal measurements.
        dut_waveform (Optional[DataTimeSignalData]): The device waveform.
        reference_waveform (Optional[DataTimeSignalData]): The reference waveform.
    """

    type: str = "PropagationDelayMeasurementData"
    measurements: ScalarMetricCollection | None = None
    dut_waveform: DataTimeSignalData | None = None
    reference_waveform: DataTimeSignalData | None = None


class PropagationDelayMeasurementDataCollection(MeasurementDataCollection):
    type: str = "PropagationDelayMeasurementDataCollection"
    collection: list[PropagationDelayMeasurementData] = []
