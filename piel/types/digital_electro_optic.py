"""
This module defines the BitPhaseMap class, which provides a mapping between bits and corresponding phases.
It also includes the necessary imports and type aliases for numerical measurement and bit measurement.
"""

import numpy as np
import pandas as pd
from .core import NumericalTypes, PielBaseModel
from .digital import BitsType


class BitPhaseMap(PielBaseModel):
    """
    A model representing a mapping between digital bits and their corresponding phases.

    Attributes:
        bits (list[AbstractBitsType] | tuple[AbstractBitsType] | np.ndarray):
            An iterable collection of bits. Can be a list, tuple, or numpy array of elements that match the AbstractBitsType.
        phase (list[NumericalTypes] | tuple[NumericalTypes] | np.ndarray):
            An iterable collection of phases. Can be a list, tuple, or numpy array of elements that match the NumericalTypes.

    Properties:
        dataframe (pd.DataFrame):
            A pandas DataFrame representation of the BitPhaseMap, combining the bits and phases into a tabular format.
    """

    bits: list[BitsType] | tuple[BitsType] | np.ndarray
    """
    bits (list[AbstractBitsType] | tuple[AbstractBitsType] | np.ndarray):
        An iterable collection of bits.
        Can be a list, tuple, or numpy array of elements that are of type AbstractBitsType.
    """

    phase: list[NumericalTypes] | tuple[NumericalTypes] | np.ndarray
    """
    phase (list[NumericalTypes] | tuple[NumericalTypes] | np.ndarray):
        An iterable collection of phases.
        Can be a list, tuple, or numpy array of elements that are of type NumericalTypes.
    """

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representation of the BitPhaseMap.

        Returns:
            pd.DataFrame: A DataFrame containing the bits and their corresponding phases.
        """
        return pd.DataFrame(self.dict())


"""
This class contains a set of unitaries that interest us to model, probably a sequence of unitaries that represent
a given simulation. It also contains a given input sequence, and a given output sequence.
"""
