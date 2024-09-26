from typing import Optional
from piel.types.connectivity.core import Instance


class MeasurementData(Instance):
    type: Optional[str] = ""


class MeasurementDataCollection(Instance):
    type: str = ""
    collection: list[MeasurementData] = []
