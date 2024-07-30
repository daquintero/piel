import skrf
from .connectivity.physical import PhysicalComponent

FrequencyNetworkModel = skrf.Network
"""
Corresponds to a container that contains a s-parameter representation.
"""


class RFPhysicalComponent(PhysicalComponent):
    network: FrequencyNetworkModel = None
