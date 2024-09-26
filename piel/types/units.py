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


A = Unit(name="ampere", datum="1", base=1, label=r"Current $A$")
dB = Unit(name="Decibel", datum="dB", base=1, label=r"Ratio $dB$")
GHz = Unit(name="Gigahertz", datum="Hertz", base=1e9, label=r"Frequency $GHz$")
Hz = Unit(name="Hertz", datum="Hertz", base=1, label=r"Frequency $Hz$")
nm = Unit(name="nanometer", datum="meter", base=1e-9, label=r"Length $nm$")
ns = Unit(name="nanosecond", datum="second", base=1e-9, label=r"Time $ns$")
mm2 = Unit(
    name="millimeter_squared", datum="meter_squared", base=1e-6, label=r"Area $mm^2$"
)
mW = Unit(name="miliwatt", datum="watt", base=1e-3, label=r"Power $mW$")
ohm = Unit(name="ohm", datum="resistance", base=1, label=r"Resistance $\Omega$")
ps = Unit(name="picosecond", datum="second", base=1e-12, label=r"Time $ps$")
ratio = Unit(name="ratio", datum="1", base=1, label=r"Ratio $u$")
s = Unit(name="second", datum="second", base=1, label=r"Time $s$")
us = Unit(name="microsecond", datum="second", base=1e-6, label=r"Time $\mu s$")
W = Unit(name="watt", datum="watt", base=1, label=r"Power $W$")
V = Unit(name="Volt", datum="voltage", base=1, label=r"Voltage $V$")

# TODO implement operational units
