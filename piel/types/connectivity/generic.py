from .physical import PhysicalConnection, PhysicalPort, PhysicalComponent
from .abstract import Port, Connection, Component, Instance

ComponentTypes = Component | PhysicalComponent
ConnectionTypes = Connection | PhysicalConnection
PortTypes = Port | PhysicalPort


class ComponentCollection(Instance):
    components: list[ComponentTypes] = []
