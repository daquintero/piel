from piel.types.metrics import ScalarMetrics
from piel.types.connectivity.physical import PhysicalComponent
from piel.types.connectivity.metrics import ComponentMetrics


class PulsedLaserMetrics(ComponentMetrics):
    """
    In this class, we define the corresponding relevant metrics to enable characterizing a modulator device.
    """

    wavelength_nm: ScalarMetrics = ScalarMetrics()
    pulse_power_W: ScalarMetrics = ScalarMetrics()
    average_power_W: ScalarMetrics = ScalarMetrics()
    pulse_repetition_rate_Hz: ScalarMetrics = ScalarMetrics()
    pulse_width_s: ScalarMetrics = ScalarMetrics()


class PulsedLaser(PhysicalComponent):
    """
    This is an equivalent model of an Electro-Optical modulator.
    """

    metrics: list[PulsedLaserMetrics] = []
