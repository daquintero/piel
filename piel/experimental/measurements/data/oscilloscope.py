from piel.types.experimental import OscilloscopeMeasurementData, OscilloscopeMeasurement


def extract_oscilloscope_data_from_measurement(
    measurement: OscilloscopeMeasurement,
) -> OscilloscopeMeasurementData:
    raise NotImplementedError(
        "This extraction method has not been generically implemented."
    )
