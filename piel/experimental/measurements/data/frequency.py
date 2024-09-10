from piel.types.experimental import (
    VNASParameterMeasurementData,
    VNASParameterMeasurement,
)


def extract_s_parameter_data_from_vna_measurement(
    measurement: VNASParameterMeasurement, **kwargs
) -> VNASParameterMeasurementData:
    import skrf

    network = skrf.Network(name=measurement.name, file=measurement.spectrum_file)
    return VNASParameterMeasurementData(
        name=measurement.name,
        network=network,
        **kwargs,
    )
