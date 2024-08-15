from piel import (
    create_all_connections,
    create_connection_list_from_ports_lists,
    create_component_connections,
)
from piel.types import (
    Port,
    PhysicalPort,
    Connection,
    PhysicalConnection,
    PhysicalComponent,
)


def test_create_all_connections():
    port1 = Port(name="port1")
    port2 = Port(name="port2")
    port3 = Port(name="port3")

    # Test default connection creation
    connections = create_all_connections([port1, port2, port3])
    assert len(connections) == 3
    assert isinstance(connections[0], Connection)
    assert connections[0].ports == (port1, port2)
    assert connections[1].ports == (port1, port3)
    assert connections[2].ports == (port2, port3)

    # Test PhysicalConnection creation
    connections = create_all_connections(
        [port1, port2, port3], connection_type_output=PhysicalConnection
    )
    assert len(connections) == 3
    # assert isinstance(connections[0], PhysicalConnection)
    # assert len(connections[0].connections) == 1
    # assert connections[0].connections[0] == (port1, port2)


def test_create_connection_list_from_ports_lists():
    port1 = Port(name="port1")
    port2 = Port(name="port2")
    port3 = Port(name="port3")
    port4 = Port(name="port4")

    connections = create_connection_list_from_ports_lists(
        [[port1, port2], [port3, port4]]
    )
    assert len(connections) == 2
    assert isinstance(connections[0], Connection)
    assert connections[0].ports == (port1, port2)
    assert connections[1].ports == (port3, port4)


def test_create_component_connections():
    component1 = PhysicalComponent(
        name="component1", ports=[PhysicalPort(name="port1")]
    )
    component2 = PhysicalComponent(
        name="component2", ports=[PhysicalPort(name="port1")]
    )

    # Test valid connections
    connections = create_component_connections(
        [component1, component2], [["component1.port1", "component2.port1"]]
    )
    assert len(connections) == 1
    assert isinstance(connections[0], Connection)
    assert connections[0].ports[0] == component1.get_port("port1")
    assert connections[0].ports[1] == component2.get_port("port1")


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
