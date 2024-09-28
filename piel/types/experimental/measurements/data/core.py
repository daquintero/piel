from typing import Optional
from piel.types.connectivity.core import Instance
from piel.base.experimental.measurements.data.core import (
    index_measurement_data_collection,
)


class MeasurementData(Instance):
    type: Optional[str] = ""


class MeasurementDataCollection(Instance):
    type: str = ""
    collection: list[MeasurementData] = []

    # Custom overwrittten methods should be defined this way.
    __getitem__ = index_measurement_data_collection
