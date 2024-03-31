from ...types import ArrayTypes

__all__ = [
    "TemperatureRangeLimitType",
    "TemperatureRangeArrayType",
    "TemperatureRangeTypes",
]

TemperatureRangeLimitType = tuple[float, float]
TemperatureRangeArrayType = ArrayTypes
TemperatureRangeTypes = TemperatureRangeLimitType | TemperatureRangeArrayType
