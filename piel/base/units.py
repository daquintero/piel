# Helper function to combine datum
def combine_datum(datum1, datum2, operation: str = "mul"):
    """
    Combine two datum based on the operation.
    For multiplication, add the exponents.
    For division, subtract the exponents.
    """
    if isinstance(datum1, str) or isinstance(datum2, str):
        # If datum is a string, handle accordingly
        if operation == "mul":
            return f"({datum1})·({datum2})"
        elif operation == "div":
            return f"({datum1})/({datum2})"
        else:
            raise ValueError("Unsupported operation for datum combination.")

    # Assuming datum1 and datum2 are dicts
    combined = datum1.copy()
    for unit, exponent in datum2.items():
        if operation == "mul":
            combined[unit] = combined.get(unit, 0) + exponent
        elif operation == "div":
            combined[unit] = combined.get(unit, 0) - exponent
        else:
            raise ValueError("Unsupported operation for datum combination.")

        # Remove unit if exponent is zero
        if combined[unit] == 0:
            del combined[unit]

    return combined


# External operator functions
def unit_mul(self, other):
    from piel.types.units import Unit

    if isinstance(other, Unit):
        # Combine base numerical factors
        new_base = self.base * other.base

        # Combine datum (unit exponents)
        new_datum = combine_datum(self.datum, other.datum, operation="mul")

        # Create a new Unit instance
        new_unit = Unit(
            name=f"({self.name} * {other.name})",
            label=f"{self.label} * {other.label}",
            shorthand=f"{self.shorthand}·{other.shorthand}",
            symbol=f"{self.symbol}·{other.symbol}",
            datum=new_datum,
            base=new_base,
        )
        return new_unit

    elif isinstance(other, (int, float)):
        # Multiply the base numerical factor
        return Unit(
            name=self.name,
            label=self.label,
            shorthand=self.shorthand,
            symbol=self.symbol,
            datum=self.datum,
            base=self.base * other,
        )
    else:
        return NotImplemented


def unit_rmul(self, other):
    # Support commutative multiplication
    return unit_mul(self, other)


def unit_add(self, other):
    from piel.types.units import Unit

    if isinstance(other, Unit):
        if self.datum != other.datum:
            raise ValueError("Cannot add Units with different dimensions.")

        # Combine base numerical factors
        new_base = self.base + other.base

        # Create a new Unit instance
        new_unit = Unit(
            name=f"({self.name} + {other.name})",
            label=f"{self.label} + {other.label}",
            shorthand=f"{self.shorthand} + {other.shorthand}",
            symbol=f"{self.symbol} + {other.symbol}",
            datum=self.datum,
            base=new_base,
        )
        return new_unit

    else:
        return NotImplemented


def unit_radd(self, other):
    # Support commutative addition
    return unit_add(self, other)


# Division
def unit_truediv(self, other):
    from piel.types.units import Unit

    if isinstance(other, Unit):
        # Combine base numerical factors
        new_base = self.base / other.base

        # Combine datum (unit exponents) with division
        new_datum = combine_datum(self.datum, other.datum, operation="div")

        # Create a new Unit instance
        new_unit = Unit(
            name=f"({self.name} / {other.name})",
            label=f"{self.label} / {other.label}",
            shorthand=f"{self.shorthand}/{other.shorthand}",
            symbol=f"{self.symbol}/{other.symbol}",
            datum=new_datum,
            base=new_base,
        )
        return new_unit

    elif isinstance(other, (int, float)):
        # Divide the base numerical factor
        return Unit(
            name=self.name,
            label=self.label,
            shorthand=self.shorthand,
            symbol=self.symbol,
            datum=self.datum,
            base=self.base / other,
        )
    else:
        return NotImplemented


def unit_rtruediv(self, other):
    from piel.types.units import Unit

    # Handle cases like 2 / meter
    if isinstance(other, (int, float)):
        new_base = other / self.base
        new_datum = combine_datum({}, self.datum, operation="div")  # Invert exponents

        # If datum is a string, adjust accordingly
        if isinstance(self.datum, str):
            new_datum = f"{other}/({self.datum})"
        else:
            # Invert the exponents in the datum dictionary
            new_datum = {unit: -exponent for unit, exponent in self.datum.items()}

        return Unit(
            name=f"({other} / {self.name})",
            label=f"{other} / {self.label}",
            shorthand=f"{other}/{self.shorthand}",
            symbol=f"{other}/{self.symbol}",
            datum=new_datum,
            base=new_base,
        )
    else:
        return NotImplemented
