"""
Note that this section of experimental types is separate from the main piel package flow because they correspond to
specific experimental files that is not yet part of the main package in a flow used as per the devices provided.
"""
import datetime
from typing import Optional
from ..types import PielBaseModel, PathTypes


class DeviceMeasurementFileMetadata(PielBaseModel):
    """
    Standard definition for a file metadata that is part of a measurement.
    """

    device_name: str
    measurement_name: str
    date: Optional[any] = datetime.time


class PropagationDelayFileCollection(DeviceMeasurementFileMetadata):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.

    The collection includes the device name, the measurement name and the date of the measurement.
    This configuration requires a device waveform, a measurement file and a reference waveform as per a propagation delay measurement.
    TODO add link to the documentation of the propagation delay measurement.
    """

    device_waveform: PathTypes
    measurement_file: PathTypes
    reference_waveform: PathTypes
    source_frequency_GHz: Optional[float]


class PropagationDelaySweepFileCollection(PielBaseModel):
    """
    This class is used to define a collection of PropagationDelayFileCollection that are part of a sweep of a parameter
    as defined within each PropagationDelayFileCollection.
    """

    sweep_parameter_name: str
    """
    The name of the parameter that is being swept. Must exist within the PropagationDelayFileCollection files definition.
    """
    files: list[PropagationDelayFileCollection]


class VNAMeasurementFileCollection(DeviceMeasurementFileMetadata):
    """
    Standard definition for a collection of files that are part of a VNA measurement.
    """

    spectrum_file: PathTypes
    source_bias_V: Optional[float]


class RFMeasurementFileCollection(
    VNAMeasurementFileCollection, PropagationDelayFileCollection
):
    pass


PropagationDelaySweepData = dict[str,]
