import pytest
from piel.experimental.types import (
    WaveformGenerator,
    WaveformGeneratorConfiguration,
    Oscilloscope,
    OscilloscopeConfiguration,
)
from piel.types import PulseSource


# Fixtures for SignalTimeSources and MinimumMaximumType
@pytest.fixture
def mock_signal_time_sources():
    # Replace this with a mock implementation of SignalTimeSources
    return PulseSource(
        voltage_1_V=0,
        voltage_2_V=5,
        delay_time_s=1e-3,
        rise_time_s=1e-6,
        fall_time_s=1e-6,
        pulse_width_s=1e-3,
        period_s=2e-3,
    )


@pytest.fixture
def mock_minimum_maximum_type():
    # Replace this with a mock implementation of MinimumMaximumType
    return (1, 100)


# Test WaveformGeneratorConfiguration
def test_waveform_generator_configuration_initialization(mock_signal_time_sources):
    config = WaveformGeneratorConfiguration(signal=mock_signal_time_sources)
    assert config.signal == mock_signal_time_sources


def test_waveform_generator_initialization(mock_signal_time_sources):
    config = WaveformGeneratorConfiguration(signal=mock_signal_time_sources)
    waveform_generator = WaveformGenerator(configuration=config)
    assert waveform_generator.configuration == config
    assert waveform_generator.configuration.signal == mock_signal_time_sources


# Test OscilloscopeConfiguration
def test_oscilloscope_configuration_initialization(mock_minimum_maximum_type):
    config = OscilloscopeConfiguration(bandwidth_Hz=mock_minimum_maximum_type)
    assert config.bandwidth_Hz == mock_minimum_maximum_type


def test_oscilloscope_initialization(mock_minimum_maximum_type):
    config = OscilloscopeConfiguration(bandwidth_Hz=mock_minimum_maximum_type)
    oscilloscope = Oscilloscope(configuration=config)
    assert oscilloscope.configuration == config
    assert oscilloscope.configuration.bandwidth_Hz == mock_minimum_maximum_type


# Test default configuration behavior for WaveformGenerator
def test_waveform_generator_default_configuration():
    waveform_generator = WaveformGenerator()
    assert waveform_generator.configuration is None


# Test default configuration behavior for Oscilloscope
def test_oscilloscope_default_configuration():
    oscilloscope = Oscilloscope()
    assert oscilloscope.configuration is None


# Add more tests as needed to verify the behavior of your classes
