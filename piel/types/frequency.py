from .connectivity.physical import PhysicalComponent
from typing import Any

FrequencyNetworkModel = Any | None  # TODO temporary nonetype till reserialization
"""
Corresponds to a container that contains a s-parameter representation.
"""


class RFPhysicalComponent(PhysicalComponent):
    network: FrequencyNetworkModel | None = None
