from typing import Optional
from ...types import PathTypes, PhysicalPort
from .device import (
    DeviceMeasurementFileMetadata,
    DeviceConfiguration,
    Device,
)


class VNAConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the VNA connectivity and configuration,
    not the experimental setup connectivity.
    """

    calibration_setting_name: Optional[str] = None
    sweep_points: int
    frequency_range_Hz: tuple[float, float]


class VNA(Device):
    """
    Represents a vector-network analyser.
    """

    configuration: Optional[VNAConfiguration] = None
    """
    Just overwrites this section of the device definition.
    """

    ports: tuple[PhysicalPort] = (
        PhysicalPort(name="PORT1", domain="RF"),
        PhysicalPort(name="PORT2", domain="RF"),
    )
    """
    Defaults to a two-port VNA, defined as PORT1 and PORT2.
    """


class VNAMeasurementFileCollection(DeviceMeasurementFileMetadata):
    """
    Standard definition for a collection of files that are part of a VNA measurement.
    """

    spectrum_file: PathTypes
    configuration: VNAConfiguration
