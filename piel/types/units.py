from typing import Literal
from piel.types.core import NumericalTypes, PielBaseModel

BaseSIUnitNameList = Literal[
    "meter",
    "second",
    "mole",
    "ampere",
    "volt",
    "kelvin",
    "candela",
    "watt",
    "dBm",
    "Hertz",
    "Decibel",
    "meter_squared",
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
Hz = Unit(name="Hertz", datum="Hertz", base=1)
dB = Unit(name="Decibel", datum="dB", base=1)
V = Unit(name="Volt", datum="V", base=1)
nm = Unit(name="nanometer", datum="meter", base=1e-9)
mm2 = Unit(name="millimeter_squared", datum="meter_squared", base=1e-6)
# TODO implement operational units
