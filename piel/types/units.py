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
    label: str = ""
    """
    Label used in plots and more.
    """


ratio = Unit(name="ratio", datum="1", base=1, label=r"Ratio $u$")
s = Unit(name="second", datum="second", base=1, label=r"Time $s$")
us = Unit(name="microsecond", datum="second", base=1e-6, label=r"Time $\mu s$")
ns = Unit(name="nanosecond", datum="second", base=1e-9, label=r"Time $ns$")
mW = Unit(name="miliwatt", datum="watt", base=1e-3, label=r"Power $mW$")
W = Unit(name="watt", datum="watt", base=1, label=r"Power $W$")
Hz = Unit(name="Hertz", datum="Hertz", base=1, label=r"Frequency $Hz$")
dB = Unit(name="Decibel", datum="dB", base=1, label=r"Ratio $dB$")
V = Unit(name="Volt", datum="V", base=1, label=r"Voltage $V$")
nm = Unit(name="nanometer", datum="meter", base=1e-9, label=r"Length $nm$")
mm2 = Unit(
    name="millimeter_squared", datum="meter_squared", base=1e-6, label=r"Area $mm^2$"
)
# TODO implement operational units
