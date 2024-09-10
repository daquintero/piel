from piel.types import PhysicalPort
from piel.types import PhysicalConnection, Connection
from piel.types import PhysicalComponent, Environment, Component


def test_physical_port_initialization():
    port = PhysicalPort()
    assert port.domain is None
    assert port.connector == ""
    assert port.manifold == ""


def test_physical_port_assignment():
    port = PhysicalPort(domain="DC", connector="USB-C", manifold="Top")
    assert port.domain == "DC"
    assert port.connector == "USB-C"
    assert port.manifold == "Top"


def test_physical_connection_initialization():
    conn = PhysicalConnection(connections=[])
    assert isinstance(conn.connections, list)


def test_physical_connection_with_components():
    conn = PhysicalConnection(connections=[Connection()], components=(Component(),))
    assert len(conn.connections) == 1
    assert isinstance(conn.components, tuple)
    assert len(conn.components) == 1


def test_physical_component_initialization():
    component = PhysicalComponent(ports=[PhysicalPort()], connections=[])
    assert isinstance(component.ports, list)
    assert isinstance(component.connections, list)
    # assert component.environment is Environment()


def test_physical_component_assignment():
    component = PhysicalComponent(
        ports=[PhysicalPort(connector="HDMI")],
        connections=[PhysicalConnection(connections=[Connection()])],
        environment=Environment(),
        manufacturer="Example Corp",
        model="Model X",
    )
    assert component.manufacturer == "Example Corp"
    assert component.model == "Model X"
    assert isinstance(component.environment, Environment)
