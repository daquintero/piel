"""
These are the corresponding prefixes from `hdl21`:

f = FEMTO = Prefix.FEMTO
p = PICO = Prefix.PICO
n = NANO = Prefix.NANO
µ = u = MICRO = Prefix.MICRO # Note both `u` and `µ` are valid
m = MILLI = Prefix.MILLI
K = KILO = Prefix.KILO
M = MEGA = Prefix.MEGA
G = GIGA = Prefix.GIGA
T = TERA = Prefix.TERA
P = PETA = Prefix.PETA
UNIT = Prefix.UNIT

"""
import hdl21 as h
import numpy as np


def convert_numeric_to_prefix(
    value: float,
):
    """
    This function converts a numeric value to a number under a SPICE unit closest to the base prefix. This allows us
    to connect a particular number real output, into a term that can be used in a SPICE netlist.
    """
    prefixes = [
        (h.Prefix.YOCTO, h.prefix.y),
        (h.Prefix.ZEPTO, h.prefix.z),
        (h.Prefix.ATTO, h.prefix.a),
        (h.Prefix.FEMTO, h.prefix.f),
        (h.Prefix.PICO, h.prefix.p),
        (h.Prefix.NANO, h.prefix.n),
        (h.Prefix.MICRO, h.prefix.µ),
        (h.Prefix.MICRO, h.prefix.u),
        (h.Prefix.MILLI, h.prefix.m),
        (h.Prefix.CENTI, h.prefix.c),
        (h.Prefix.DECI, h.prefix.d),
        # (Prefix.UNIT, ''),
        (h.Prefix.DECA, h.prefix.D),
        (h.Prefix.KILO, h.prefix.K),
        (h.Prefix.MEGA, h.prefix.M),
        (h.Prefix.GIGA, h.prefix.G),
        (h.Prefix.TERA, h.prefix.T),
        (h.Prefix.PETA, h.prefix.P),
        (h.Prefix.EXA, h.prefix.E),
        (h.Prefix.ZETTA, h.prefix.Z),
        (h.Prefix.YOTTA, h.prefix.Y),
    ]

    base_10 = np.log10(value)
    value_target_base = np.floor(base_10)

    closest_prefix = None
    min_difference = 2
    for prefix, _ in prefixes:
        difference = abs(value_target_base - prefix.value)
        if difference < min_difference:
            min_difference = difference
            closest_prefix = prefix

    value /= 10**closest_prefix.value

    return value * closest_prefix
