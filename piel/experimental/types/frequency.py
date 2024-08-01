from typing import Optional
from ...types import PhysicalPort
from .device import (
    DeviceConfiguration,
    Device,
)
from .measurements.generic import FrequencyMeasurementConfigurationTypes


class VNAConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the VNA connectivity and configuration,
    not the experimental setup connectivity.
    """

    calibration_setting_name: Optional[str] = None
    measurement_configuration: FrequencyMeasurementConfigurationTypes = None


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
