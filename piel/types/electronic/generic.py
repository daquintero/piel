from piel.types.electronic.hva import PowerAmplifier
from piel.types.electronic.lna import LowNoiseTwoPortAmplifier
from piel.types.electronic.core import RFElectronicCircuit
from piel.types.connectivity.generic import ComponentCollection

RFAmplifierTypes = PowerAmplifier | LowNoiseTwoPortAmplifier | RFElectronicCircuit


class RFAmplifierCollection(ComponentCollection):
    components: list[RFAmplifierTypes] = []
