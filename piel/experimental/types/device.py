"""
Note that this section of experimental types is separate from the main piel package flow because they correspond to
specific experimental files that is not yet part of the main package in a flow used as per the devices provided.
"""
from typing import Optional
from ...types import PielBaseModel, PhysicalComponent, PathTypes


class DeviceMeasurement(PielBaseModel):
    """
    Standard definition for a measurement.
    """

    name: Optional[str] = None
    parent_directory: Optional[PathTypes] = None


class DeviceConfiguration(PielBaseModel):
    pass


class Device(PhysicalComponent):
    """
    Corresponds to the abstraction of a given device measurement.
    """

    configuration: Optional[DeviceConfiguration] = None

    manufacturer: Optional[str] = None
    """
    The manufacturer of the device.
    """


class MeasurementDevice(Device):
    measurement: Optional[DeviceMeasurement] = None
    """
    Contains the measurement information.
    """
