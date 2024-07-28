from typing import Optional
from .....types import Instance
from ....types import ExperimentInstance


class MeasurementData(Instance):
    experimental_instance: Optional[ExperimentInstance] = None
