"""
Note that this section of experimental types is separate from the main piel package flow because they correspond to
specific experimental files that is not yet part of the main package in a flow used as per the devices provided.
"""

from typing import Optional
from ...types import PielBaseModel, PhysicalComponent
from .measurements.core import MeasurementConfiguration


class DeviceConfiguration(PielBaseModel):
    pass


class Device(PhysicalComponent):
    """
    Corresponds to the abstraction of a given device measurement.
    """

    configuration: Optional[DeviceConfiguration] = None
    serial_number: str = ""


class MeasurementDevice(Device):
    measurement: Optional[MeasurementConfiguration] = None
    """
    Contains the measurement information.
    """
