from piel.types.signal.frequency.core import Phasor
from piel.types.core import ArrayTypes, NumericalTypes

PhasorTypes = Phasor | NumericalTypes | ArrayTypes
"""
Different ways to represent frequency-domain information. In principle, both array and individual representations of
complex numbers which can represent phasors. These could be phasor-arrays, as phasor components.

Contains a very clear notation to translate with a complex number that is a full phasor-representation.

.. math::

     Ae^{i(\omega t+\theta)}
"""
