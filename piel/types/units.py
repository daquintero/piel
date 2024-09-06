from typing import Literal
from piel.types.core import NumericalTypes, PielBaseModel

BaseSIUnitNameList = Literal[
    "meter", "second", "mole", "ampere", "volt", "kelvin", "candela", "watt", "dBm"
]


class Unit(PielBaseModel):
    name: str = ""
    datum: BaseSIUnitNameList | str
    base: NumericalTypes = 1
    """In the format 1eN"""


ratio = Unit(name="ratio", datum="1", base=1)
s = Unit(name="second", datum="second", base=1)
us = Unit(name="microsecond", datum="second", base=1e-6)
ns = Unit(name="nanosecond", datum="second", base=1e-9)
mW = Unit(name="miliwatt", datum="watt", base=1e-3)
W = Unit(name="watt", datum="watt", base=1)
