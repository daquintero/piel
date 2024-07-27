from .physical import PhysicalConnection, PhysicalPort, PhysicalComponent
from .abstract import Port, Connection, Component

ComponentTypes = Component | PhysicalComponent
ConnectionTypes = Connection | PhysicalConnection
PortTypes = Port | PhysicalPort
