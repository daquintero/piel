from __future__ import annotations
from piel.types.core import PathTypes
from piel.types.connectivity.abstract import Instance


class MeasurementConfiguration(Instance):
    """
    Standard definition for a measurement configuration.
    """

    name: str = ""
    parent_directory: PathTypes = ""
    measurement_type: str = ""


class Measurement(Instance):
    """
    Standard definition for a measurement. This should be the container for all the measurement files, it is not the data container.
    """

    name: str = ""
    type: str = ""
    parent_directory: PathTypes = ""


class MeasurementCollection(Instance):
    """
    Generic class for MeasurementCollection
    """

    type: str = ""
    collection: list[Measurement] = []
