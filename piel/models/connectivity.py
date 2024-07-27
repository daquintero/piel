from typing import Optional, Callable
from ..types import Port, Connection, PhysicalConnection, ConnectionTypes


def create_all_connections(
    ports: list[Port],
    connection_factory: Optional[Callable[[list[Port]], Connection]] = None,
    connection_type_output: Optional[ConnectionTypes] = Connection,
) -> list[ConnectionTypes]:
    """
    This function receives a list of ports and creates the connections between them as two-port relationships.
    It returns a list of connections. More than two ports can be provided, and it will create all the possible connections.

    Parameters
    ----------
    ports : list[Port]
        The ports list to create connections.

    connection_factory : Optional[Callable[[list[Port]], Connection]], optional
        A function that creates a connection object from a list of ports.
        The function should receive a list of ports and return a connection object.
        If not provided, a default connection factory will be used.
        The default connection factory creates a tuple of ports as a connection object.
        The default is None.

    connection_type_output : Optional[type[Connection]], optional
        The type of connection object to return.

        If not provided, the default connection factory will be used.

        The default is None

    Returns
    -------
    list[Connection]
        A list of connections that were created.
    """

    def default_connection_factory(ports: list[Port, Port]) -> ConnectionTypes:
        """
        This function creates a connection object from a list of ports, by default it creates a tuple of ports as a
        connection object. Should be two ports.
        """
        raw_connection = tuple(ports)

        if type(connection_type_output) is type(Connection):
            connection = Connection(ports=raw_connection)
        elif type(connection_type_output) is type(PhysicalConnection):
            connection = PhysicalConnection(connections=[raw_connection])
        else:
            raise TypeError(
                f"Expected a Connection or PhysicalConnection type, got {type(connection_type_output)} instead."
            )
        return connection

    connection_factory = connection_factory or default_connection_factory

    # Ensure that ports is a list
    if not isinstance(ports, list):
        # Raise descriptive error
        raise TypeError(f"Expected a list of ports, got {type(ports)} instead.")

    connections = []
    for i, port1 in enumerate(ports):
        for port2 in ports[i + 1 :]:
            # Verify port2 is a port
            if not isinstance(port2, Port):
                # Raise descriptive error
                raise TypeError(f"Expected a port, got {type(port2)} instead.")
            else:
                connections.append(connection_factory([port1, port2]))

    return connections


def create_connection_list_from_ports_lists(
    port_connection_list: list[list[Port]],
) -> list[ConnectionTypes]:
    """
    When a list of a list of ports is provided, we construct all the required connections accordingly. TODO more docs.
    """
    connection_list = list()
    for raw_connection_i in port_connection_list:
        connection_list_i = create_all_connections(raw_connection_i)
        connection_list.extend(connection_list_i)

    return connection_list
