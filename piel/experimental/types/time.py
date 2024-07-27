from ...types import SignalTimeSources
from .device import DeviceConfiguration, Device


class WaveformGeneratorConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the WaveformGenerator connectivity and configuration,
    not the experimental setup connectivity.
    """

    signal: SignalTimeSources
    """
    Contains an instantiation of the signal configuration applied as a reference.
    """


class WaveformGenerator(Device):
    """
    Represents a vector-network analyser.
    """

    configuration: WaveformGeneratorConfiguration
    """
    Just overwrites this section of the device definition.
    """


class OscilloscopeConfiguration(DeviceConfiguration):
    """
    This class corresponds to the configuration of data which is just inherent to the Oscilloscope connectivity and configuration,
    not the experimental setup connectivity.
    """


class Oscilloscope(Device):
    """
    Represents an oscilloscope
    """

    configuration: OscilloscopeConfiguration
    """
    Just overwrites this section of the device definition.
    """
