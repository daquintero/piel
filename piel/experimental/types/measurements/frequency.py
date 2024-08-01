from ....types import PathTypes
from .core import MeasurementConfiguration, Measurement


class VNASParameterMeasurementConfiguration(MeasurementConfiguration):
    frequency_range_Hz: tuple[float, float] = None
    sweep_points: int = None
    test_port_power_dBm: float = None


class VNAPowerSweepMeasurementConfiguration(MeasurementConfiguration):
    base_frequency_Hz: float = None
    power_range_dBm: tuple[float, float] = None


class VNASParameterMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of a S-Parameter VNA measurement.
    """

    spectrum_file: PathTypes


class VNAPowerSweepMeasurement(Measurement):
    """
    Standard definition for a collection of files that are part of a VNA measurement.
    """

    spectrum_file: PathTypes


VNASParameterMeasurementCollection = list[VNASParameterMeasurement]
VNAPowerSweepMeasurementCollection = list[VNASParameterMeasurementCollection]
