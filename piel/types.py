"""
We create a set of parameters that can be used throughout the project for optimisation.

The numerical solver is jax and is imported throughout the module.
"""
import os
import pathlib
import pydantic
import types
import numpy as np
import jax.numpy as jnp

__all__ = [
    "ArrayTypes",
    "PathTypes",
    "PielBaseModel",
]

PathTypes = str | pathlib.Path | os.PathLike | types.ModuleType
ArrayTypes = np.ndarray | jnp.ndarray


class PielBaseModel(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def supplied_parameters(self):
        # This method returns a list of parameter names that have been supplied (i.e., are not None)
        return [param for param, value in self.__dict__.items() if value is not None]


class QuantityType(PielBaseModel):
    """
    The base class for all cable types.
    """

    units: str
    """
    The units of the type.
    """
