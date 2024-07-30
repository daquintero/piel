from typing import Optional, Callable
from ....types import PCB, PhysicalPort, PhysicalConnection, Connection

from typing import List, Tuple


def create_pcb(
    pcb_name: str = None,
    port_name_list: List[str] = None,
    connection_tuple_list: List[Tuple[str, str]] = None,
    port_factory: Optional[Callable[[str], PhysicalPort]] = None,
    **kwargs,
) -> PCB:
    """
    Defines a PCB component instantiation.

    Args:
        port_name_list (List[str], optional): The list of port names. Defaults to None.
        connection_tuple_list (List[Tuple[str, str]], optional): The list of connections between ports. Defaults to None.
        port_factory (Optional[Callable[[str], PhysicalPort]], optional): The factory function to create the ports.
            Needs to contain the port definition parameters. Defaults to None.
    """

    if pcb_name is None:
        pcb_name = "pcb"

    def default_port_factory(name: str, pcb_name: str) -> PhysicalPort:
        return PhysicalPort(name=name, parent_component_name=pcb_name)

    # Default arguments if None are provided
    if port_name_list is None:
        port_name_list = []
    if connection_tuple_list is None:
        connection_tuple_list = []
    if port_factory is None:
        port_factory = default_port_factory

    # Create ports based on port_name_list
    ports_list = [port_factory(name=port_name) for port_name in port_name_list]

    # Create a dictionary to quickly lookup ports by name
    port_dict = {port.name: port for port in ports_list}

    # Create connections
    connections = []
    for connection_tuple in connection_tuple_list:
        port1_name, port2_name = connection_tuple
        if port1_name not in port_dict:
            raise ValueError(f"Port '{port1_name}' not found in the port list.")
        if port2_name not in port_dict:
            raise ValueError(f"Port '{port2_name}' not found in the port list.")

        port1 = port_dict[port1_name]
        port2 = port_dict[port2_name]
        connection = Connection(ports=(port1, port2))
        connections.append(
            PhysicalConnection(
                connections=[connection],
            ),
        )

    return PCB(name=pcb_name, ports=ports_list, connections=connections, **kwargs)
