"""
Note that this section of experimental types is separate from the main piel package flow because they correspond to
specific experimental files that is not yet part of the main package in a flow used as per the devices provided.
"""
from typing import Optional
from ...types import PielBaseModel, PhysicalComponent


class DeviceMeasurementFileMetadata(PielBaseModel):
    """
    Standard definition for a file metadata that is part of a measurement.
    """

    device_name: str
    measurement_name: str
    date: Optional[any] = None


class DeviceConfiguration(PielBaseModel):
    pass


class Device(PhysicalComponent):
    """
    Corresponds to the abstraction of a given device measurement.
    """

    configuration: Optional[DeviceConfiguration] = None
