from piel.types.experimental import (
    VNASParameterMeasurementData,
    VNASParameterMeasurementDataCollection,
    VNAPowerSweepMeasurementData,
    MeasurementData,
    MeasurementDataCollection,
)


# Test VNASParameterMeasurementData
def test_vna_sparameter_measurement_data_initialization():
    measurement_data = VNASParameterMeasurementData()

    assert isinstance(measurement_data, MeasurementData)
    assert measurement_data.type == "VNASParameterMeasurementData"
    assert measurement_data.network is None


def test_vna_sparameter_measurement_data_default_initialization():
    measurement_data = VNASParameterMeasurementData()

    assert isinstance(measurement_data, MeasurementData)
    assert measurement_data.type == "VNASParameterMeasurementData"
    assert measurement_data.network is None


# Test VNAPowerSweepMeasurementData
def test_vna_power_sweep_measurement_data_initialization():
    network = None
    measurement_data = VNAPowerSweepMeasurementData(network=network)

    assert isinstance(measurement_data, MeasurementData)
    assert measurement_data.type == "VNAPowerSweepMeasurementData"
    assert measurement_data.network is None


def test_vna_power_sweep_measurement_data_default_initialization():
    measurement_data = VNAPowerSweepMeasurementData()

    assert isinstance(measurement_data, MeasurementData)
    assert measurement_data.type == "VNAPowerSweepMeasurementData"
    assert measurement_data.network is None


# Test VNASParameterMeasurementDataCollection
def test_vna_sparameter_measurement_data_collection_initialization():
    measurement_data_1 = VNASParameterMeasurementData()
    measurement_data_2 = VNASParameterMeasurementData()

    data_collection = VNASParameterMeasurementDataCollection(
        collection=[measurement_data_1, measurement_data_2]
    )

    assert isinstance(data_collection, MeasurementDataCollection)
    assert data_collection.type == "VNASParameterMeasurementDataCollection"
    assert data_collection.collection == [measurement_data_1, measurement_data_2]


def test_vna_sparameter_measurement_data_collection_default_initialization():
    data_collection = VNASParameterMeasurementDataCollection()

    assert isinstance(data_collection, MeasurementDataCollection)
    assert data_collection.type == "VNASParameterMeasurementDataCollection"
    assert data_collection.collection == []


# Test FrequencyMeasurementDataCollection with mixed measurement
def test_frequency_measurement_data_collection_initialization():
    sparam_data = VNASParameterMeasurementData()
    power_sweep_data = VNAPowerSweepMeasurementData()

    data_collection = VNASParameterMeasurementDataCollection()

    assert isinstance(data_collection, MeasurementDataCollection)
    assert data_collection.type == "VNASParameterMeasurementDataCollection"
    assert data_collection.collection == []


def test_frequency_measurement_data_collection_default_initialization():
    data_collection = VNASParameterMeasurementDataCollection()

    assert isinstance(data_collection, MeasurementDataCollection)
    assert data_collection.type == "VNASParameterMeasurementDataCollection"
    assert data_collection.collection == []


# Add more tests as needed for additional methods, edge cases, and behaviors.
