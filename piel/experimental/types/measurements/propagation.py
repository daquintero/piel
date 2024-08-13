from ....types import PathTypes
from .core import MeasurementConfiguration, Measurement, MeasurementCollection


class PropagationDelayMeasurementConfiguration(MeasurementConfiguration):
    measurement_configuration_type: str = "PropagationDelayMeasurementConfiguration"
    pass


class PropagationDelayMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.
    The collection includes the device name, the measurement name and the date of the measurement.
    This configuration requires a device waveform, a measurement file and a reference waveform as per a propagation delay measurement.
    TODO add link to the documentation of the propagation delay measurement.
    """

    type: str = "PropagationDelayMeasurement"
    dut_waveform_file: PathTypes = ""
    reference_waveform_file: PathTypes = ""
    measurements_file: PathTypes = ""


class PropagationDelayMeasurementCollection(MeasurementCollection):
    type: str = "PropagationDelayMeasurementCollection"
    collection: list[PropagationDelayMeasurement] = []
