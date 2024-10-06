from piel.types.core import PielBaseModel
from piel.types.metrics import ScalarMetric
from piel.types.signal.frequency.generic import PhasorTypes


class FrequencyMetric(PielBaseModel):
    """
    Creates a mapping between a given phasor input and a given scalar metrics response.
    """

    phasor: PhasorTypes | None = None
    metric: ScalarMetric


class FrequencyMetricCollection(PielBaseModel):
    metrics: list[FrequencyMetric] = []
