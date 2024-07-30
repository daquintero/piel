from typing import Optional
from .device import Device, DeviceConfiguration


class SourcemeterConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Sourcemeter connectivity and configuration,
    not the experimental setup connectivity.
    """

    voltage_set_V: Optional[float] = None
    voltage_range_V: Optional[tuple] = None
    current_limit_A: Optional[float] = None
    voltage_limit_V: Optional[float] = None


class Sourcemeter(Device):
    """
    Represents a sourcemeter.
    """

    configuration: Optional[SourcemeterConfiguration] = None


class MultimeterConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Multimeter connectivity and configuration,
    """


class Multimeter(Device):
    """
    Represents a multimeter.
    """
