import skrf
from ...types import VNASParameterMeasurementData, VNASParameterMeasurement


def extract_s_parameter_data_from_vna_measurement(
    measurement: VNASParameterMeasurement, **kwargs
) -> VNASParameterMeasurementData:
    network = skrf.Network(name=measurement.name, file=measurement.spectrum_file)
    return VNASParameterMeasurementData(
        name=measurement.name,
        network=network,
        **kwargs,
    )
