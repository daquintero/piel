# External operator functions for Quantity

# Addition
def quantity_add(self, other):
    from piel.types.quantity import Quantity

    if not isinstance(other, Quantity):
        return NotImplemented

    if self.unit.datum != other.unit.datum:
        raise ValueError("Cannot add Quantities with different units.")

    # Add the values
    new_value = self.value + other.value

    # Assume units are the same; you might want to handle unit conversion here
    new_quantity = Quantity(
        name=f"({self.name} + {other.name})",
        label=f"{self.label} + {other.label}",
        shorthand=f"{self.shorthand} + {other.shorthand}",
        symbol=f"{self.symbol} + {other.symbol}",
        value=new_value,
        unit=self.unit,
    )
    return new_quantity


def quantity_radd(self, other):
    return quantity_add(self, other)


# Subtraction
def quantity_sub(self, other):
    from piel.types.quantity import Quantity

    if not isinstance(other, Quantity):
        return NotImplemented

    if self.unit.datum != other.unit.datum:
        raise ValueError("Cannot subtract Quantities with different units.")

    # Subtract the values
    new_value = self.value - other.value

    # Assume units are the same; you might want to handle unit conversion here
    new_quantity = Quantity(
        name=f"({self.name} - {other.name})",
        label=f"{self.label} - {other.label}",
        shorthand=f"{self.shorthand} - {other.shorthand}",
        symbol=f"{self.symbol} - {other.symbol}",
        value=new_value,
        unit=self.unit,
    )
    return new_quantity


def quantity_rsub(self, other):
    from piel.types.quantity import Quantity

    if not isinstance(other, Quantity):
        return NotImplemented
    # For rsub, perform other - self
    if self.unit.datum != other.unit.datum:
        raise ValueError("Cannot subtract Quantities with different units.")

    new_value = other.value - self.value

    new_quantity = Quantity(
        name=f"({other.name} - {self.name})",
        label=f"{other.label} - {self.label}",
        shorthand=f"{other.shorthand} - {self.shorthand}",
        symbol=f"{other.symbol} - {self.symbol}",
        value=new_value,
        unit=other.unit,
    )
    return new_quantity


# Multiplication
def quantity_mul(self, other):
    from piel.types.quantity import Quantity
    from piel.types.units import Unit

    if isinstance(other, Quantity):
        # Multiply values
        new_value = self.value * other.value
        # Multiply units
        new_unit = self.unit * other.unit

        new_quantity = Quantity(
            name=f"({self.name} * {other.name})",
            label=f"{self.label} * {other.label}",
            shorthand=f"{self.shorthand}·{other.shorthand}",
            symbol=f"{self.symbol}·{other.symbol}",
            value=new_value,
            unit=new_unit,
        )
        return new_quantity

    elif isinstance(other, Unit):
        # Multiply value by 1 (unit multiplication)
        new_value = self.value
        # Multiply units
        new_unit = self.unit * other

        new_quantity = Quantity(
            name=f"({self.name} * {other.name})",
            label=f"{self.label} * {other.label}",
            shorthand=f"{self.shorthand}·{other.shorthand}",
            symbol=f"{self.symbol}·{other.symbol}",
            value=new_value,
            unit=new_unit,
        )
        return new_quantity

    elif isinstance(other, (int, float)):
        # Scale the value
        new_value = self.value * other
        new_quantity = Quantity(
            name=self.name,
            label=self.label,
            shorthand=self.shorthand,
            symbol=self.symbol,
            value=new_value,
            unit=self.unit,
        )
        return new_quantity

    else:
        return NotImplemented


def quantity_rmul(self, other):
    # Support commutative multiplication
    return quantity_mul(self, other)


# Division
def quantity_truediv(self, other):
    from piel.types.quantity import Quantity
    from piel.types.units import Unit

    if isinstance(other, Quantity):
        # Divide values
        new_value = self.value / other.value
        # Divide units
        new_unit = self.unit / other.unit

        new_quantity = Quantity(
            name=f"({self.name} / {other.name})",
            label=f"{self.label} / {other.label}",
            shorthand=f"{self.shorthand}/{other.shorthand}",
            symbol=f"{self.symbol}/{other.symbol}",
            value=new_value,
            unit=new_unit,
        )
        return new_quantity

    elif isinstance(other, Unit):
        # Divide unit from quantity's unit
        new_value = self.value
        new_unit = self.unit / other

        new_quantity = Quantity(
            name=f"({self.name} / {other.name})",
            label=f"{self.label} / {other.label}",
            shorthand=f"{self.shorthand}/{other.shorthand}",
            symbol=f"{self.symbol}/{other.symbol}",
            value=new_value,
            unit=new_unit,
        )
        return new_quantity

    elif isinstance(other, (int, float)):
        # Scale the value
        new_value = self.value / other
        new_quantity = Quantity(
            name=self.name,
            label=self.label,
            shorthand=f"{self.shorthand}/{other}",
            symbol=f"{self.symbol}/{other}",
            value=new_value,
            unit=self.unit,
        )
        return new_quantity

    else:
        return NotImplemented


def quantity_rtruediv(self, other):
    from piel.types.quantity import Quantity
    from piel.types.units import Unit

    # Handle cases like 2 / Quantity
    if isinstance(other, (int, float)):
        new_value = other / self.value
        # Invert the unit
        if isinstance(self.unit.datum, dict):
            new_unit = Unit(
                name=f"({other} / {self.unit.name})",
                label=f"{other} / {self.unit.label}",
                shorthand=f"{other}/{self.unit.shorthand}",
                symbol=f"{other}/{self.unit.symbol}",
                datum={unit: -exponent for unit, exponent in self.unit.datum.items()},
                base=other / self.unit.base,
            )
        else:
            # If datum is a string, handle accordingly
            new_unit = Unit(
                name=f"({other} / {self.unit.name})",
                label=f"{other} / {self.unit.label}",
                shorthand=f"{other}/{self.unit.shorthand}",
                symbol=f"{other}/{self.unit.symbol}",
                datum=f"{other}/({self.unit.datum})",
                base=other / self.unit.base,
            )

        new_quantity = Quantity(
            name=f"({other} / {self.name})",
            label=f"{other} / {self.label}",
            shorthand=f"{other}/{self.shorthand}",
            symbol=f"{other}/{self.symbol}",
            value=new_value,
            unit=new_unit,
        )
        return new_quantity
    else:
        return NotImplemented
