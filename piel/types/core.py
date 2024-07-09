import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import pathlib
import pydantic
import types
from typing import Optional, Literal

# Type aliases for various path and array types used throughout the module.
PathTypes = str | pathlib.Path | os.PathLike | types.ModuleType
ArrayTypes = np.ndarray | jnp.ndarray
NumericalTypes = int | float | np.dtype | jnp.dtype
TupleFloatType = tuple[float, ...]
TupleIntType = tuple[int, ...]
TupleNumericalType = tuple[NumericalTypes, ...]
PackageArrayType = Literal["qutip", "jax", "numpy", "list", "tuple"] | TupleIntType


class PielBaseModel(pydantic.BaseModel):
    """
    A base model class that serves as the foundation for other models in the project.
    This class extends pydantic's BaseModel and includes additional configuration
    settings for strict validation and immutability.

    Attributes:
        Config: A nested class to configure the behavior of the model.
            arbitrary_types_allowed (bool): Allows arbitrary types.
            extra (str): Forbids extra attributes not defined in the model.
            validate_assignment (bool): Validates fields on assignment.
            strict (bool): Enforces strict type validation.
            allow_mutation (bool): Prevents mutation of the model's fields.

    Methods:
        supplied_parameters() -> list[str]:
            Returns a list of parameter names that have been supplied (i.e., are not None).
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_assignment = True
        strict = True

    def supplied_parameters(self):
        """
        Returns a list of parameter names that have been supplied (i.e., are not None).

        Returns:
            list[str]: A list of parameter names with non-None values.
        """
        return [param for param, value in self.__dict__.items() if value is not None]


class QuantityType(PielBaseModel):
    """
    A model representing a quantity with optional units.

    Attributes:
        units (Optional[str]): The units of the quantity.
    """

    units: Optional[str] = None
    """
    The units of the type.
    """
