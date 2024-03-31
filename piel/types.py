"""
We create a set of parameters that can be used throughout the project for optimisation.

The numerical solver is jax and is imported throughout the module.
"""
import os
import pathlib
import types
import numpy as np
import jax.numpy as jnp

__all__ = [
    "ArrayTypes",
    "PathTypes",
]

PathTypes = str | pathlib.Path | os.PathLike | types.ModuleType
ArrayTypes = np.ndarray | jnp.ndarray
