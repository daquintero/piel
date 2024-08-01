from ...types import MeasurementCollectionTypes, MeasurementDataCollectionTypes
from ..map import measurement_to_data_map, measurement_to_data_method_map
def extract_data_from_measurement_collection(
    measurement_collection: MeasurementCollectionTypes,
    measurement_to_data_map: dict = measurement_to_data_map,
    measurement_to_data_method_map: dict = measurement_to_data_method_map,
) -> MeasurementDataCollectionTypes:
    """
    The goal of this function is to compose the data from a collection of measurement references.
    Based on each type of measurement, it will apply an extraction function based on the data mapping accordingly.
    It will return a collection of data types which is inherent to the type of the measurement collection provided.
    """
    data_collection: MeasurementDataCollectionTypes = list()

    for measurement_i in measurement_collection:
        # Identify correct data mapping
        measurement_data_type = measurement_to_data_map[measurement_i.__class__.__name__]
        extract_data_method = measurement_to_data_method_map[measurement_i.__class__.__name__]
        measurement_data_i = extract_data_method(measurement_i)
        assert isinstance(measurement_data_i, measurement_data_type)
        data_collection.append(measurement_data_i)

    return data_collection
