"""
We create a set of parameters that can be used throughout the project for optimisation.

The numerical solver is normally delegated for as `numpy` but there are cases where a much faster solver is desired, and where different functioanlity is required. For example, `sax` uses `JAX` for its numerical solver. In this case, we will create a global numerical solver that we can use throughout the project, and that can be extended and solved accordingly for the particular project requirements.
"""
import pathlib
import sys
import types

__all__ = [
    "numerical_solver",
    "nso",
    "piel_path_types",
]


if "jax" in sys.modules:
    import jax.numpy as jnp

    numerical_solver = jnp
elif "numpy" in sys.modules:
    import numpy

    numerical_solver = numpy
else:
    import numpy

    numerical_solver = numpy

nso = numerical_solver
piel_path_types = str | pathlib.Path | types.ModuleType
