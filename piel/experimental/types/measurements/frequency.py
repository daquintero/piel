from ....types import PathTypes
from .core import MeasurementConfiguration, Measurement, MeasurementCollection


class VNASParameterMeasurementConfiguration(MeasurementConfiguration):
    measurement_configuration_type: str = "VNASParameterMeasurementConfiguration"
    frequency_range_Hz: tuple[float, float] = None
    sweep_points: int = None
    test_port_power_dBm: float = None


class VNAPowerSweepMeasurementConfiguration(MeasurementConfiguration):
    measurement_configuration_type: str = "VNAPowerSweepMeasurementConfiguration"
    base_frequency_Hz: float = None
    power_range_dBm: tuple[float, float] = None


class VNASParameterMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of a S-Parameter VNA measurement.
    """

    type: str = "VNASParameterMeasurement"
    spectrum_file: PathTypes = None


class VNAPowerSweepMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of a VNA measurement.
    """

    type: str = "VNAPowerSweepMeasurement"
    spectrum_file: PathTypes = None


class VNASParameterMeasurementCollection(MeasurementCollection):
    type: str = "VNASParameterMeasurementCollection"
    collection: list[VNASParameterMeasurement] = []


class VNAPowerSweepMeasurementCollection(MeasurementCollection):
    type: str = "VNAPowerSweepMeasurementCollection"
    collection: list[VNASParameterMeasurementCollection] = []
