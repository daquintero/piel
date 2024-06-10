"""
This module defines type aliases for representing temperature ranges in various formats, using core array types.
"""

from .core import ArrayTypes

# Type alias for a tuple representing temperature range limits as floats.
TemperatureRangeLimitType = tuple[float, float]
"""
TemperatureRangeLimitType:
    A tuple representing the lower and upper limits of a temperature range.
    Each element in the tuple is a float indicating a specific temperature value.
    Example: (min_temperature, max_temperature)
"""

# Type alias for temperature ranges represented as arrays.
TemperatureRangeArrayType = ArrayTypes
"""
TemperatureRangeArrayType:
    An array type (either numpy or jax array) representing a range of temperatures.
    This is used for more detailed or discrete temperature data points.
"""

# Type alias for representing temperature ranges, either as a tuple of limits or as an array of values.
TemperatureRangeTypes = TemperatureRangeLimitType | TemperatureRangeArrayType
"""
TemperatureRangeTypes:
    A union type that can represent temperature ranges in two formats:
    - TemperatureRangeLimitType: A tuple of floats defining the lower and upper limits of the range.
    - TemperatureRangeArrayType: An array (numpy or jax) containing a series of temperature values.
"""
