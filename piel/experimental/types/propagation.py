from ...types import PielBaseModel, PathTypes
from .device import DeviceMeasurementFileMetadata


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


class PropagationDelaySweepFileCollection(PielBaseModel):
    """
    This class is used to define a collection of PropagationDelayFileCollection that are part of a sweep of a parameter
    as defined within each PropagationDelayFileCollection.
    """

    files: list[PropagationDelayFileCollection]
