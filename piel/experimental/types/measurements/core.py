from ....types import PielBaseModel, PathTypes


class MeasurementConfiguration(PielBaseModel):
    """
    Standard definition for a measurement configuration.
    """

    name: str = ""
    parent_directory: PathTypes = None
    measurement_type: str = ""


class Measurement(PielBaseModel):
    """
    Standard definition for a measurement. This should be the container for all the measurement files, it is not the data container.
    """

    name: str = ""
    type: str = ""
    parent_directory: PathTypes = None


class MeasurementCollection(PielBaseModel):
    """
    Generic class for MeasurementCollection
    """

    type: str = ""
    collection: list[Measurement] = []
