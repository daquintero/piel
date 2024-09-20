from piel.types.environment import Environment
from piel.types.connectivity.abstract import Instance
from piel.types.reference import Reference


class ComponentMetrics(Instance):
    """
    Note that a given metrics needs to be matched to a given environment in which the measurements are performed.
    For example, room temperature metrics are not equivalent to cryogenic metrics measurements, and as such
    require a collection of variables.
    """

    reference: Reference = Reference()
    environment: Environment = Environment()
