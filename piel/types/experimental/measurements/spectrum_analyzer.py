from piel.types.core import PathTypes
from piel.types.experimental.measurements.core import (
    MeasurementConfiguration,
    Measurement,
    MeasurementCollection
)


class SpectrumMeasurementConfiguration(MeasurementConfiguration):
    measurement_configuration_type: str = "SpectrumMeasurementConfiguration"


class SpectrumMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of an either Electrical or Optical Spectrum measurement.
    """

    type: str = "SpectrumMeasurement"
    spectrum_file: PathTypes = ""


class SpectrumMeasurementCollection(MeasurementCollection):
    type: str = "SpectrumMeasurementCollection"
    collection: list[SpectrumMeasurement] = []

