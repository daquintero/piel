from ....types import PielBaseModel, PathTypes


class MeasurementConfiguration(PielBaseModel):
    """
    Standard definition for a measurement configuration.
    """

    name: str = None
    parent_directory: PathTypes = None
    measurement_type: str = None


class Measurement(PielBaseModel):
    """
    Standard definition for a measurement. This should be the container for all the measurement files, it is not the data container.
    """

    name: str = None
    parent_directory: PathTypes = None
