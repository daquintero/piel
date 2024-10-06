from piel.types.metrics import ScalarMetric
from piel.types.connectivity.physical import PhysicalComponent
from piel.types.connectivity.metrics import ComponentMetrics


class PulsedLaserMetrics(ComponentMetrics):
    """
    In this class, we define the corresponding relevant metrics to enable characterizing a modulator device.
    """

    wavelength_nm: ScalarMetric = ScalarMetric()
    pulse_power_W: ScalarMetric = ScalarMetric()
    average_power_W: ScalarMetric = ScalarMetric()
    pulse_repetition_rate_Hz: ScalarMetric = ScalarMetric()
    pulse_width_s: ScalarMetric = ScalarMetric()


class PulsedLaser(PhysicalComponent):
    """
    This is an equivalent model of an Electro-Optical modulator.
    """

    metrics: list[PulsedLaserMetrics] = []
