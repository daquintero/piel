from piel.types.core import PathTypes
from .core import MeasurementConfiguration, Measurement, MeasurementCollection


class OscilloscopeMeasurementConfiguration(MeasurementConfiguration):
    measurement_configuration_type: str = "OscilloscopeMeasurementConfiguration"
    pass


class OscilloscopeMeasurement(Measurement):
    """
    Generic, extensible OscilloscopeMeasurement
    """

    type: str = "OscilloscopeMeasurement"
    waveform_file_list: list[PathTypes] = []
    measurements_file: PathTypes = ""


class OscilloscopeMeasurementCollection(MeasurementCollection):
    type: str = "OscilloscopeMeasurementCollection"
    collection: list[OscilloscopeMeasurement] = []
