from piel.types.connectivity.abstract import Instance
from piel.types.reference import Reference


class ComponentMetrics(Instance):
    reference: Reference | None = None
