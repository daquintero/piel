from typing import Optional
from ....types import Instance, PathTypes, PielBaseModel
from ..device import DeviceMeasurement
from ..experiment import ExperimentInstance


class PropagationDelayMeasurement(DeviceMeasurement):
    """
    Standard definition for a collection of files that are part of a propagation delay measurement.
    The collection includes the device name, the measurement name and the date of the measurement.
    This configuration requires a device waveform, a measurement file and a reference waveform as per a propagation delay measurement.
    TODO add link to the documentation of the propagation delay measurement.
    """
    experiment_instance: ExperimentInstance
    dut_waveform_prefix: PathTypes
    reference_waveform_prefix: PathTypes
    measurement_file_prefix: PathTypes


class PropagationDelayMeasurementSweep(Instance):
    """
    This class is used to define a collection of PropagationDelayFileCollection that are part of a sweep of a parameter
    as defined within each PropagationDelayFileCollection.
    """
    measurements: list[PropagationDelayMeasurement]
