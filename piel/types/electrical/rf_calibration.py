from ...types import PhysicalComponent, PhysicalPort


class Short(PhysicalComponent):
    ports: list[PhysicalPort] = [PhysicalPort(name="SHORT", connector="SMA_3.5mm", manifold="82052D")]


class Open(PhysicalComponent):
    ports: list[PhysicalPort] = [PhysicalPort(name="OPEN", connector="SMA_3.5mm", manifold="82052D")]


class Load(PhysicalComponent):
    ports: list[PhysicalPort] = [PhysicalPort(name="LOAD", connector="SMA_3.5mm", manifold="82052D")]


class Through(PhysicalComponent):
    ports: list[PhysicalPort] = [
        PhysicalPort(name="IN", connector="SMA_3.5mm", manifold="82052D"),
        PhysicalPort(name="OUT", connector="SMA_3.5mm", manifold="82052D"),
    ]
