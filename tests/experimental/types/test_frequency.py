import pytest
from piel.experimental.types import (
    VNA,
    VNAConfiguration,
    VNASParameterMeasurementConfiguration,
)
from piel.types import PhysicalPort


# Fixtures for FrequencyMeasurementConfigurationTypes and PhysicalPort
@pytest.fixture
def mock_frequency_measurement_configuration():
    # Replace this with a mock implementation of FrequencyMeasurementConfigurationTypes
    return VNASParameterMeasurementConfiguration()


@pytest.fixture
def mock_physical_ports():
    return (
        PhysicalPort(name="PORT1", domain="RF"),
        PhysicalPort(name="PORT2", domain="RF"),
    )


# Test VNAConfiguration
def test_vna_configuration_initialization(mock_frequency_measurement_configuration):
    config = VNAConfiguration(
        calibration_setting_name="Calibration1",
        measurement_configuration=mock_frequency_measurement_configuration,
    )
    assert config.calibration_setting_name == "Calibration1"
    assert config.measurement_configuration == mock_frequency_measurement_configuration


def test_vna_configuration_default_initialization():
    config = VNAConfiguration()
    assert config.calibration_setting_name == ""
    assert config.measurement_configuration is None


# Test VNA
def test_vna_initialization_with_configuration(
    mock_frequency_measurement_configuration,
):
    config = VNAConfiguration(
        calibration_setting_name="Calibration1",
        measurement_configuration=mock_frequency_measurement_configuration,
    )
    vna = VNA(configuration=config)
    assert vna.configuration == config
    assert vna.configuration.calibration_setting_name == "Calibration1"
    assert (
        vna.configuration.measurement_configuration
        == mock_frequency_measurement_configuration
    )


def test_vna_default_configuration():
    vna = VNA()
    assert vna.configuration is None


def test_vna_ports_initialization(mock_physical_ports):
    vna = VNA()
    assert isinstance(vna.ports, tuple) or isinstance(vna.ports, list)
    assert len(vna.ports) == 2
    assert vna.ports[0].name == "PORT1"
    assert vna.ports[0].domain == "RF"
    assert vna.ports[1].name == "PORT2"
    assert vna.ports[1].domain == "RF"


# Add more tests as needed to verify the behavior of your classes
