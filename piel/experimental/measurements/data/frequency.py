import skrf
from skrf.io.touchstone import hfss_touchstone_2_network
from ...types import VNASParameterMeasurementData, VNASParameterMeasurement

def extract_s_parameter_data_from_vna_measurement(
    measurement: VNASParameterMeasurement
) -> VNASParameterMeasurementData:
    network = hfss_touchstone_2_network(measurement.spectrum_file)
    return VNASParameterMeasurementData(
        network=network
    )

