from piel.types.connectivity.abstract import Component
from piel.types.environment import Environment


class TemperatureStage(Environment, Component):
    """
    Standard definition for a generic temperature stage.
    """

    surface_area_m2: float = None  # TODO move to a geometry type.


class Cryostat(Component):
    temperature_stages: list[TemperatureStage] = []
