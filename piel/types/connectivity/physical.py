from typing import Optional
from .abstract import Port, Connection, Component
from ..core import PielBaseModel
from ..environment import Environment
from ..signal.core import ElectricalSignalDomains


class PhysicalPort(Port):
    """
    This refers to any form of physical port which can be connected to. The socket refers to the corresponding physical location where such a port is connected to.
    """

    domain: Optional[ElectricalSignalDomains] = None
    connector: Optional[str] = None
    manifold: Optional[str] = None


class PhysicalConnection(PielBaseModel):
    """
    Describes a set of physical ports which are all connected. Represents a physical connection between interfaces.

    The components represent the physical implementation of the connections for the same connection index.
    """

    connections: tuple[Connection]
    components: Optional[tuple[Component]]


class PhysicalComponent(Component):
    """
    Represents the data of a physical component or device.
    """

    ports: list[PhysicalPort]
    connections: Optional[list[PhysicalConnection]] = None
    environment: Optional[Environment] = None
