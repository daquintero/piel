"""
This module defines type aliases for components and files structures used in photonic circuit design and simulation.
It includes type definitions for connection, S-parameter matrices, and photonic circuit components.
"""

from typing import Any
from .core import ArrayTypes
from piel.types.connectivity.generic import ConnectionTypes


# Type alias for an S-parameter matrix, which includes a matrix of array measurement and a tuple of port names.
SParameterMatrixTuple = tuple[ArrayTypes, ConnectionTypes]
"""
SParameterMatrixTuple:
    A tuple representing an S-parameter matrix used in circuit simulations.
    It includes:
    - ArrayTypes: A matrix (numpy or jax array) representing the S-parameters.
    - ConnectionTypes: A tuple of strings representing the corresponding port names.
"""

# Type alias for a callable representing an optical transmission circuit in SAX.
OpticalTransmissionCircuit = Any  # sax.saxtypes.Callable
"""
OpticalTransmissionCircuit:
    A callable type representing an optical transmission circuit in the SAX framework.
    This is used for functions that model the behavior of optical circuits.
"""

# Type alias for a recursive netlist in SAX, which describes a hierarchical circuit structure.
RecursiveNetlist = Any  # sax.RecursiveNetlist
"""
RecursiveNetlist:
    A type representing a recursive netlist in the SAX framework.
    This type is used to describe the hierarchical structure of photonic circuits.
"""

# Type alias for a photonic circuit component in gdsfactory.
PhotonicCircuitComponent = Any  # gf.Component
"""
PhotonicCircuitComponent:
    A type representing a component in a photonic circuit, as defined in the gdsfactory framework.
    This type is used to handle and manipulate photonic components in circuit designs.
"""
