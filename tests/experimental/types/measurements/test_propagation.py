from piel.experimental.types import (
    PropagationDelayMeasurementConfiguration,
    PropagationDelayMeasurement,
    PropagationDelayMeasurementCollection,
)


# Test PropagationDelayMeasurementConfiguration
def test_propagation_delay_measurement_configuration_initialization():
    config = PropagationDelayMeasurementConfiguration()
    assert (
        config.measurement_configuration_type
        == "PropagationDelayMeasurementConfiguration"
    )


# Test PropagationDelayMeasurement
def test_propagation_delay_measurement_initialization():
    dut_waveform = "/path/to/dut_waveform"
    reference_waveform = "/path/to/reference_waveform"
    measurements_file = "/path/to/measurements_file"

    measurement = PropagationDelayMeasurement(
        dut_waveform_file=dut_waveform,
        reference_waveform_file=reference_waveform,
        measurements_file=measurements_file,
    )

    assert measurement.type == "PropagationDelayMeasurement"
    assert measurement.dut_waveform_file == dut_waveform
    assert measurement.reference_waveform_file == reference_waveform
    assert measurement.measurements_file == measurements_file


def test_propagation_delay_measurement_default_initialization():
    measurement = PropagationDelayMeasurement()

    assert measurement.type == "PropagationDelayMeasurement"
    assert measurement.dut_waveform_file == ""
    assert measurement.reference_waveform_file == ""
    assert measurement.measurements_file == ""


# Test PropagationDelayMeasurementCollection
def test_propagation_delay_measurement_collection_initialization():
    measurement_1 = PropagationDelayMeasurement()
    measurement_2 = PropagationDelayMeasurement()

    collection = PropagationDelayMeasurementCollection(
        collection=[measurement_1, measurement_2]
    )

    assert collection.type == "PropagationDelayMeasurementCollection"
    assert collection.collection == [measurement_1, measurement_2]


def test_propagation_delay_measurement_collection_default_initialization():
    collection = PropagationDelayMeasurementCollection()

    assert collection.type == "PropagationDelayMeasurementCollection"
    assert collection.collection == []


# Add more tests as needed for additional methods, edge cases, and behaviors.
