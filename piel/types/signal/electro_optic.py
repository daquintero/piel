from piel.types.connectivity.abstract import Instance
from piel.types.signal.dc_data import SignalDC
from piel.types.signal.frequency.transmission import PathTransmission


class ElectroOpticDCPathTransmission(Instance):
    """
    It is not logically possible to decouple the output signal from the ports at which this is measured, due to the
    vectorial and non-scalar nature of the electro-optic network relationship.
    """

    input_dc: SignalDC
    output: PathTransmission


class ElectroOpticDCNetworkTransmission(Instance):
    """
    Applicable when you have multiple inputs and multiple paths
    """

    path_transmission_list: list[ElectroOpticDCPathTransmission] = []
