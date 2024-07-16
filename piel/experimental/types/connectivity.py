from typing import Optional
from ...types import Port, Connection, Component, PielBaseModel, ElectricalSignalDomains
from piel.types.environment import Environment


class PhysicalPort(Port):
    """
    This refers to any form of physical port which can be connected to. The socket refers to the corresponding physical location where such a port is connected to.
    """

    domain: Optional[ElectricalSignalDomains] = None
    connector: Optional[str] = None
    manifold: Optional[str] = None


class PhysicalConnection(PielBaseModel):
    """
    Describes a set of physical ports which are are all connected. Represents a physical connection between interfaces.

    The components represent the physical implementation of the connections for the same connection index.
    """

    connections: tuple[Connection]
    components: tuple[Component]


class PhysicalComponent(Component):
    """
    Represents the data of a physical component or device.
    """

    ports: list[PhysicalPort]
    connections: list[PhysicalConnection]
    environment: Optional[Environment] = None
