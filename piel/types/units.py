from typing import Literal
from piel.types.core import NumericalTypes, PielBaseModel

BaseSIUnitNameList = Literal[
    "meter", "second", "mole", "ampere", "volt", "kelvin", "candela"
]


class Unit(PielBaseModel):
    name: str = ""
    datum: BaseSIUnitNameList | str
    base: NumericalTypes = 1
    """In the format 1eN"""


s = Unit(name="second", datum="second", base=1)
us = Unit(name="microsecond", datum="second", base=1e-6)
ns = Unit(name="nanosecond", datum="second", base=1e-9)
