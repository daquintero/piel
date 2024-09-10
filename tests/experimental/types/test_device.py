from piel.types.experimental import (
    DeviceConfiguration,
    Device,
    MeasurementDevice,
    MeasurementConfiguration,
)
from piel.types import PhysicalComponent, PielBaseModel


# Test DeviceConfiguration
def test_device_configuration_initialization():
    config = DeviceConfiguration()
    assert isinstance(config, PielBaseModel)


# Test Device
def test_device_initialization():
    device = Device()
    assert isinstance(device, PhysicalComponent)
    assert device.configuration is None


def test_device_initialization_with_configuration():
    config = DeviceConfiguration()
    device = Device(configuration=config)
    assert device.configuration == config


# Test MeasurementDevice
def test_measurement_device_initialization():
    measurement_device = MeasurementDevice()
    assert isinstance(measurement_device, Device)
    assert measurement_device.configuration is None
    assert measurement_device.measurement is None


def test_measurement_device_initialization_with_measurement():
    measurement_config = MeasurementConfiguration()
    measurement_device = MeasurementDevice(measurement=measurement_config)
    assert measurement_device.measurement == measurement_config


def test_measurement_device_initialization_with_configuration_and_measurement():
    config = DeviceConfiguration()
    measurement_config = MeasurementConfiguration()
    measurement_device = MeasurementDevice(
        configuration=config, measurement=measurement_config
    )
    assert measurement_device.configuration == config
    assert measurement_device.measurement == measurement_config


# Add more tests as needed for additional methods and edge cases
