from ...types import PhysicalPort, PulseSource
from ..types import WaveformGenerator, WaveformGeneratorConfiguration


def create_one_port_square_wave_waveform_generator(
    peak_to_peak_voltage_V: float,
    rise_time_s: float,
    fall_time_s: float,
    frequency_Hz: float,
):
    period_s = 1 / frequency_Hz

    # TODO configure this properly
    # Configure a square wave signal
    signal = PulseSource(
        voltage_1_V=0,
        voltage_2_V=peak_to_peak_voltage_V,
        delay_time_s=0,
        rise_time_s=rise_time_s,
        fall_time_s=fall_time_s,
        pulse_width_s=period_s / 2,
        period_s=period_s,
    )

    # Configure the waveform generator
    configuration = WaveformGeneratorConfiguration(signal=signal)

    # Configure the ports
    ports = [
        PhysicalPort(
            name="CH1",
            domain="RF",
            connector="SMA",
        ),
    ]

    return WaveformGenerator(
        name="two_port_oscilloscope",
        ports=ports,
        configuration=configuration,
        manufacturer="Tektronix",
    )


def AWG70001A(**kwargs) -> WaveformGenerator:
    pass
