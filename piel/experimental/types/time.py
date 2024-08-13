from typing import Optional
from ...types import SignalTimeSources, MinimumMaximumType
from .device import DeviceConfiguration, Device, MeasurementDevice


class WaveformGeneratorConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the WaveformGenerator connectivity and configuration,
    not the experimental setup connectivity.
    """

    signal: SignalTimeSources = None
    """
    Contains an instantiation of the signal configuration applied as a reference.
    """


class WaveformGenerator(Device):
    """
    Represents a vector-network analyser.
    """

    configuration: WaveformGeneratorConfiguration = None
    """
    Just overwrites this section of the device definition.
    """


class OscilloscopeConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Oscilloscope connectivity and configuration,
    not the experimental setup connectivity.
    """

    bandwidth_Hz: MinimumMaximumType = None


class Oscilloscope(MeasurementDevice):
    """
    Represents an oscilloscope
    """

    configuration: Optional[OscilloscopeConfiguration] = None
    """
    Just overwrites this section of the device definition.
    """
