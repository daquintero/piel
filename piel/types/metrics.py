from piel.types.connectivity.abstract import Instance
from piel.types.core import NumericalTypes
from piel.types.units import Unit, ratio


class ScalarMetrics(Instance):
    """
    Standard definition for a scalar metrics. It includes the value, mean, min, max, standard deviation and count.
    """

    value: NumericalTypes | None = None
    mean: NumericalTypes | None = None
    min: NumericalTypes | None = None
    max: NumericalTypes | None = None
    standard_deviation: NumericalTypes | None = None
    unit: Unit = ratio
