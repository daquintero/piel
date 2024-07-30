from ...types import PhysicalComponent, MinimumMaximumType


class RFComponent(PhysicalComponent):
    bandwidth: MinimumMaximumType = None


class PowerSplitter(RFComponent):
    pass


class BiasTee(RFComponent):
    pass


class Attenuator(RFComponent):
    nominal_attenuation_dB: float = None
    pass
