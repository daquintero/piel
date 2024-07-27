from typing import Optional, Callable
from ....types import PCB, PhysicalPort, PhysicalConnection


def create_pcb(
    port_name_list: list[str] = None,
    connection_tuple_list: list[tuple[str, str]] = None,
    port_factory: Optional[Callable] = None,
    **kwargs
) -> PCB:
    """
    Defines a PCB component instantiation.

    Args:
        port_name_list (list[str], optional): The list of port names. Defaults to None
        connection_tuple_list (list[tuple[str, str]], optional): The list of connections between ports. Defaults to None.
        port_factory (Optional[Callable], optional): The factory function to create the ports.
            Needs to contain the port definition parameters. Defaults to None.
    """

    def default_port_factory(name: str) -> PhysicalPort:
        return PhysicalPort(name=name)

    ports_list = list()

    if port_factory is None:
        port_factory = default_port_factory

    if connection_tuple_list is None:
        connection_tuple_list = list()

    for port_name in port_name_list:
        ports_list.append(
            port_factory(name=port_name),
        )

    connections = list()
    for connection_tuple in connection_tuple_list:
        # Create a connection between two ports, note that we need to find the ports in the previously defined list.
        port1 = next(port for port in ports_list if port.name == connection_tuple[0])
        port2 = next(port for port in ports_list if port.name == connection_tuple[1])
        connections.append(
            PhysicalConnection(
                connections=(
                    port1,
                    port2,
                ),
            ),
        )

    return PCB(ports=ports_list, connections=connections, **kwargs)
