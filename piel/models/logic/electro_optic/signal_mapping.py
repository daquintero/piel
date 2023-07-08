"""
In this function we implement different methods of mapping electronic signals to phase.

One particular implementation of phase mapping would be:

.. list-table:: Example Basic Phase Mapping
   :header-rows: 1

   * - Bit
     - Phase
   * - b0
     - :math:`\\phi_0 \\to 0`
   * - b1
     - :math:`\\phi_1 \\to \\pi`

We can define the two corresponding angles that this would be.

A more complex implementation of phase mapping can be similar to a DAC mapping: a bitstring within a converter bit-size can map directly to a particular phase space within a particular mapping.
"""
import numpy as np
import pandas as pd

__all__ = [
    "bits_array_from_bits_amount",
    "linear_bit_phase_map",
    "return_phase_array_from_data_series",
]


def bits_array_from_bits_amount(bits_amount: int) -> np.ndarray:
    """
    Returns an array of bits from a given amount of bits.

    Args:
        bits_amount(int): Amount of bits to generate.

    Returns:
        bit_array(np.ndarray): Array of bits.
    """
    maximum_integer_represented = 2 ** (bits_amount)
    int_array = np.arange(maximum_integer_represented)
    bit_array = np.vectorize(np.base_repr)(int_array)
    return bit_array


def linear_bit_phase_map(
    bits_amount: int,
    final_phase_rad: float,
    initial_phase_rad: float = 0,
    return_dataframe: bool = True,
    quantization_error: float = 0.000001,
) -> dict | pd.DataFrame:
    """
    Returns a linear direct mapping of bits to phase.

    Args:
        bits_amount(int): Amount of bits to generate.
        final_phase_rad(float): Final phase to map to.
        initial_phase_rad(float): Initial phase to map to.

    Returns:
        bit_phase_mapping(dict): Mapping of bits to phase.
    """
    bits_array = bits_array_from_bits_amount(bits_amount)
    phase_division_amount = len(bits_array) - 1
    phase_division_step = (
        final_phase_rad - initial_phase_rad
    ) / phase_division_amount - quantization_error
    linear_phase_array = np.arange(
        initial_phase_rad, final_phase_rad, phase_division_step
    )
    bit_phase_mapping_raw = {
        "bits": bits_array,
        "phase": linear_phase_array,
    }
    if return_dataframe:
        bit_phase_mapping = pd.DataFrame(bit_phase_mapping_raw)
    else:
        bit_phase_mapping = bit_phase_mapping_raw
    return bit_phase_mapping


def return_phase_array_from_data_series(
    data_series: pd.Series,
    phase_map: pd.DataFrame | pd.Series,
) -> list:
    """
    Returns a list of phases from a given data series and phase map.
    # TODO optimise lookup table speed

    Args:
        data_series(pd.Series): Data series to map.
        phase_map(pd.DataFrame | pd.Series): Phase map to use.

    Returns:
        phase_array(list): List of phases.
    """
    phase_array = []
    for code_i in data_series.values:
        phase = phase_map[phase_map.bits == str(code_i)].phase.values[0]
        phase_array.append(phase)
    return phase_array
