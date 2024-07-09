"""
This module defines type aliases for components and files structures used in photonic circuit design and simulation.
It includes type definitions for ports, S-parameter matrices, and photonic circuit components.
"""

import gdsfactory as gf
import sax
from .core import ArrayTypes

# Type alias for a tuple of port names as strings.
PortsTuple = tuple[str, ...]
"""
PortsTuple:
    A tuple representing the names of ports in a photonic circuit.
    Each element in the tuple is a string corresponding to a port name.
"""

# Type alias for an S-parameter matrix, which includes a matrix of array types and a tuple of port names.
SParameterMatrixTuple = tuple[ArrayTypes, PortsTuple]
"""
SParameterMatrixTuple:
    A tuple representing an S-parameter matrix used in circuit simulations.
    It includes:
    - ArrayTypes: A matrix (numpy or jax array) representing the S-parameters.
    - PortsTuple: A tuple of strings representing the corresponding port names.
"""

# Type alias for a callable representing an optical transmission circuit in SAX.
OpticalTransmissionCircuit = sax.saxtypes.Callable
"""
OpticalTransmissionCircuit:
    A callable type representing an optical transmission circuit in the SAX framework.
    This is used for functions that model the behavior of optical circuits.
"""

# Type alias for a recursive netlist in SAX, which describes a hierarchical circuit structure.
RecursiveNetlist = sax.RecursiveNetlist
"""
RecursiveNetlist:
    A type representing a recursive netlist in the SAX framework.
    This type is used to describe the hierarchical structure of photonic circuits.
"""

# Type alias for a photonic circuit component in gdsfactory.
PhotonicCircuitComponent = gf.Component
"""
PhotonicCircuitComponent:
    A type representing a component in a photonic circuit, as defined in the gdsfactory framework.
    This type is used to handle and manipulate photonic components in circuit designs.
"""
