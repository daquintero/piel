from typing import Optional, Callable
from piel.types import (
    Port,
    Connection,
    PhysicalConnection,
    ConnectionTypes,
    ComponentTypes,
)


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


def create_component_connections(
    components: list[ComponentTypes],
    connection_reference_str_list: list[str] | list[list[str]],
) -> list[ConnectionTypes]:
    """
    The way this function works is by composing the ports namespaces from the names of the components,
    and a given ports dot notation which corresponds to that component.

    Notes
    -----

    The dot notation would be in the format ``"component_1.port1"``. Hence, the input to a connection would be
    ``["component1.port1", "component2.port1"]`` and this function would compile into generating the corresponding
    ports. This is by splitting the component name and port name accordingly and then programmatically acquiring
    the corresponding `Port` reference and creating the `Connection` from this.

    Parameters
    ----------

    components : list[ComponentTypes]
        The components to create connections from.
    connection_reference_str_list : list[str] | list[list[str]]
        The list of strings that represent the connections to create.

    Returns
    -------
    list[ConnectionTypes]
        The list of connections created from the components.
    """
    # Verify the list has at least an element inside it
    if len(connection_reference_str_list) == 0:
        raise ValueError(
            "The list of connection references must have at least one connection."
        )

    # Ensure connection_reference_str_list is a list of lists
    if isinstance(connection_reference_str_list[0], str):
        connection_reference_str_list = [connection_reference_str_list]

    connection_list = []

    for connection_reference in connection_reference_str_list:
        # Ensure each connection reference is a list of two strings
        # TODO extend to more than two strings
        if not isinstance(connection_reference, list) or len(connection_reference) != 2:
            raise ValueError("Each connection reference must be a list of two strings.")

        # Split the connection reference strings into component and port names
        component1_name, port1_name = connection_reference[0].split(".")
        component2_name, port2_name = connection_reference[1].split(".")

        # Initialize ports
        port1 = None
        port2 = None

        # Get the port references
        for component in components:
            if component.name == component1_name:
                port1 = component.get_port(port1_name)
            if component.name == component2_name:
                port2 = component.get_port(port2_name)

        # Check if the ports were found
        if port1 is None or port2 is None:
            raise ValueError(
                f"Could not find the ports for the connection {connection_reference}"
            )

        # Create the connection
        connection = Connection(ports=(port1, port2))
        connection_list.append(connection)

    return connection_list
