"""
The following is taken from sax. The reason is: piel wants to have the closest integration possible with sax,
but not core-depend on it.
Also, sax is a great project and Floris is also great, so hoping this
supports their functionality and any users integration with piel system design tools.
SAX Types and type coercions
"""

from typing import Dict, Tuple, Union

from jaxtyping import Array as Array
from jaxtyping import Complex as Complex
from jaxtyping import Int as Int

IntArray1D = Int[Array, " dim"]
""" One dimensional integer array """

FloatArray1D = Complex[Array, " dim"]
""" One dimensional float array """

ComplexArray1D = Complex[Array, " dim"]
""" One dimensional complex array """

IntArrayND = Int[Array, "..."]
""" N-dimensional integer array """

FloatArrayND = Complex[Array, "..."]
""" N-dimensional float array """

ComplexArrayND = Complex[Array, "..."]
""" N-dimensional complex array """

PortMap = Dict[str, int]
""" A mapping from a port name (str) to a port index (int) """

PortCombination = Tuple[str, str]
""" A combination of two port names (str, str) """

SDict = Dict[PortCombination, ComplexArrayND]
""" A mapping from a port combination to an S-parameter or an array of S-parameters
Equivalent to a PathTransmission definition.

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDense = Tuple[ComplexArrayND, PortMap]
""" A dense S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array
combined with a port map. If (N+2)-D array the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCoo = Tuple[IntArray1D, IntArray1D, ComplexArrayND, PortMap]
""" A sparse S-matrix in COO format (recommended for internal library use only)

An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three arrays and a port map.
The three arrays represent the input port indices [`int`], output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse matrix.
The port map maps a port name [`str`] to a port index [`int`]. Only these four arrays **together** and in this specific **order** are considered a valid `SCoo` representation!

Example:

.. code-block::

    Si = jnp.arange(3, dtype=int)
    Sj = jnp.array([0, 1, 0], dtype=int)
    Sx = jnp.array([3.0, 4.0, 1.0])
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    scoo: sax.SCoo = (Si, Sj, Sx, port_map)

"""

SType = Union[SDict, SCoo, SDense]
""" An SDict, SDense or SCOO """
