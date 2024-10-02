from __future__ import annotations

from typing import Optional
from .core import Instance
from .timing import TimeMetricsTypes, ZeroTimeMetrics
from .metrics import ComponentMetrics


class Port(Instance):
    """
    This represents the fundamental data structure to identify a port.
    """

    parent_component_name: str = ""


class Connection(Instance):
    """
    This represents the fundamental data structure to identify a connection between two ports.

    Note that any connection has a
    """

    ports: tuple[Port, Port] | list[Port] = tuple()
    time: TimeMetricsTypes = ZeroTimeMetrics


class Component(Instance):
    """
    This represents the fundamental data structure to identify a component with ports and internal or external connectivity.
    """

    ports: list[Port] = []

    connections: list[Connection] = []

    components: list[Component] = []
    """
    Note the recursive relationship that a component can be composed of multiple components.
    """

    metrics: list[ComponentMetrics] = []
    """
    Note that a given component might have a set of metrics corresponding to multiple variations of the testing conditions.
    """

    def get_port(self, port_name: str) -> Optional[Port]:
        """
        Get a port by its name.
        """
        port_dict = {port.name: port for port in self.ports if port.name is not None}
        return port_dict.get(port_name, None)


# Type alias for a tuple of port names as strings.
PortMap = tuple[str, ...] | tuple[Port, ...] | tuple[int, ...]
"""
PortMap:
    A tuple representing the names of ports in a photonic circuit.
    Each element in the tuple is a string corresponding to a port name.
"""
