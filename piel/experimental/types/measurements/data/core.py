from .....types import Instance


class MeasurementData(Instance):
    type: str = ""


class MeasurementDataCollection(Instance):
    type: str = ""
    collection: list[MeasurementData] = []
