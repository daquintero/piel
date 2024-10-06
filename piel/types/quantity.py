from piel.types.core import NumericalTypes
from piel.types.symbolic import SymbolicValue
from piel.types.units import Unit, ratio
from piel.base.quantity import (
    quantity_add,
    quantity_mul,
    quantity_sub,
    quantity_rtruediv,
    quantity_truediv,
    quantity_radd,
    quantity_rmul,
    quantity_rsub,
)


class Quantity(SymbolicValue):
    name: str = ""
    value: NumericalTypes | None = None
    unit: Unit = ratio

    __add__ = quantity_add
    __mul__ = quantity_mul
    __sub__ = quantity_sub
    __radd__ = quantity_radd
    __rmul__ = quantity_rmul
    __rsub__ = quantity_rsub
    __truediv__ = quantity_truediv
    __rtruediv__ = quantity_rtruediv
    __iadd__ = quantity_add
    __imul__ = quantity_mul
    __isub__ = quantity_sub
