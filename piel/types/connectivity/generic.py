from .physical import PhysicalConnection, PhysicalPort, PhysicalComponent
from .abstract import Port, Connection, Component, Instance

ComponentTypes = Component | PhysicalComponent
ConnectionTypes = (
    Connection
    | PhysicalConnection
    | tuple[str, ...]
    | tuple[Port, ...]
    | tuple[int, ...]
)
PortTypes = Port | PhysicalPort | str | int


class ComponentCollection(Instance):
    components: list[ComponentTypes] = []
