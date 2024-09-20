from piel.types.connectivity.physical import PhysicalComponent
from piel.types.connectivity.metrics import ComponentMetrics


class ElectroOpticModulatorMetrics(ComponentMetrics):
    """
    In this class, we define the corresponding relevant metrics to enable characterizing a modulator device.
    """

    pass


class ElectroOpticModulator(PhysicalComponent):
    """
    This is an equivalent model of an Electro-Optical modulator.
    """

    metrics: list[ElectroOpticModulatorMetrics] = []
