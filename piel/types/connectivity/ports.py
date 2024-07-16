from typing import Optional
from piel.types import PielBaseModel


class Port(PielBaseModel):
    name: str


Connection = tuple[Port, Port]


class Component(PielBaseModel):
    """
    This represents the fundamental data structure to identify a component with ports and internal or external connectivity.
    """

    name: Optional[str] | None = None
    ports: list[Port]
    connections: list[Connection]
