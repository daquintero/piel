"""
We create a set of parameters that can be used throughout the project for optimisation.

The numerical solver is jax and is imported throughout the module.
"""
import pathlib
import types

__all__ = [
    "piel_path_types",
]

piel_path_types = str | pathlib.Path | types.ModuleType
