import pytest
from piel.models.physical.electrical import (
    create_one_port_square_wave_waveform_generator,
    AWG70001A,
)
from piel.types.experimental import (
    WaveformGenerator,
    WaveformGeneratorConfiguration,
)
from piel.types import PhysicalPort, PulseSource


def test_create_one_port_square_wave_waveform_generator():
    peak_to_peak_voltage_V = 5.0
    rise_time_s = 1e-9
    fall_time_s = 1e-9
    frequency_Hz = 1e6

    generator = create_one_port_square_wave_waveform_generator(
        peak_to_peak_voltage_V=peak_to_peak_voltage_V,
        rise_time_s=rise_time_s,
        fall_time_s=fall_time_s,
        frequency_Hz=frequency_Hz,
    )

    assert isinstance(generator, WaveformGenerator)
    assert generator.name == "two_port_oscilloscope"
    assert len(generator.ports) == 1

    port = generator.ports[0]
    assert isinstance(port, PhysicalPort)
    assert port.name == "CH1"
    assert port.domain == "RF"
    assert port.connector == "SMA"

    configuration = generator.configuration
    assert isinstance(configuration, WaveformGeneratorConfiguration)

    signal = configuration.signal
    assert isinstance(signal, PulseSource)
    assert signal.voltage_1_V == 0
    assert signal.voltage_2_V == peak_to_peak_voltage_V
    assert signal.rise_time_s == rise_time_s
    assert signal.fall_time_s == fall_time_s
    assert signal.period_s == pytest.approx(1 / frequency_Hz)
    assert signal.pulse_width_s == pytest.approx(1 / (2 * frequency_Hz))


def test_awg70001a():
    signal = PulseSource(
        voltage_1_V=0,
        voltage_2_V=5,
        delay_time_s=1e-3,
        rise_time_s=1e-6,
        fall_time_s=1e-6,
        pulse_width_s=1e-3,
        period_s=2e-3,
    )  # Replace with appropriate initialization if needed

    generator = AWG70001A(signal=signal)

    assert isinstance(generator, WaveformGenerator)
    assert generator.name == "AWG70001A"
    assert generator.manufacturer == "Tektronix"
    assert len(generator.ports) == 1

    port = generator.ports[0]
    assert isinstance(port, PhysicalPort)
    assert port.name == "CH1"
    assert port.domain == "RF"
    assert port.connector == "SMA"

    configuration = generator.configuration
    assert isinstance(configuration, WaveformGeneratorConfiguration)
    assert configuration.signal == signal


def test_awg70001a_with_custom_name():
    signal = PulseSource(
        voltage_1_V=0,
        voltage_2_V=5,
        delay_time_s=1e-3,
        rise_time_s=1e-6,
        fall_time_s=1e-6,
        pulse_width_s=1e-3,
        period_s=2e-3,
    )

    generator = AWG70001A(signal=signal)


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
