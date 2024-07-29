from .device import Device, DeviceConfiguration


class SourcemeterConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Sourcemeter connectivity and configuration,
    not the experimental setup connectivity.
    """

    current_limit_A: float
    voltage_limit_V: float


class Sourcemeter(Device):
    """
    Represents a sourcemeter.
    """

    configuration: SourcemeterConfiguration


class MultimeterConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Multimeter connectivity and configuration,
    """


class Multimeter(Device):
    """
    Represents a multimeter.
    """
