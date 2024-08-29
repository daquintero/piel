from ...types import PropagationDelayMeasurementData, PropagationDelayMeasurement


def extract_propagation_delay_data_from_measurement(
    measurement: PropagationDelayMeasurement,
) -> PropagationDelayMeasurementData:
    raise NotImplementedError(
        "This generic data extraction method has not been generically implemented."
    )
