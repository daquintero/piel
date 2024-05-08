"""
TODO implement this function.
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

A more complex implementation of phase mapping can be similar to a DAC mapping: a bitstring within a converter
bit-size can map directly to a particular phase space within a particular mapping."""
import numpy as np
import jax.numpy as jnp  # TODO add typing
import pandas as pd
from typing import Callable, Optional, Iterable, Literal
from ....integration.type_conversion import array_types, convert_array_type, absolute_to_threshold, tuple_int_type
from ....tools.sax.utils import sax_to_s_parameters_standard_matrix
from ....tools.qutip.fock import fock_states_only_individual_modes
from ....types import ArrayTypes, NumericalTypes
from .types import FockStatePhaseTransitionType

__all__ = [
    "bits_array_from_bits_amount",
    "convert_phase_array_to_bit_array",
    "extract_phase",
    "find_nearest_bit_for_phase",
    "format_electro_optic_fock_transition",
    "get_state_phase_transitions",
    "get_state_to_phase_map",
    "linear_bit_phase_map",
    "return_phase_array_from_data_series",
]


def bits_array_from_bits_amount(
    bits_amount: int,
    bit_format: Literal["int", "str"] = "int",
) -> np.ndarray:
    """
    Returns an array of bits from a given amount of bits.

    Args:
        bits_amount(int): Amount of bits to generate.
        bit_format(str): Format of the bits to generate.

    Returns:
        bit_array(np.ndarray): Array of bits.
    """
    maximum_integer_represented = 2 ** (bits_amount)
    int_array = np.arange(maximum_integer_represented)
    bit_array = np.vectorize(np.base_repr)(int_array)
    if bit_format == "int":
        pass
    elif bit_format == "str":
        # Add leading zeros to bit strings
        bit_array = np.vectorize(lambda x: x.zfill(bits_amount))(bit_array)
    return bit_array


def convert_phase_array_to_bit_array(
    phase_array: Iterable,
    phase_bit_dataframe: pd.DataFrame,
    phase_series_name: str = "phase",
    bit_series_name: str = "bit",
    rounding_function: Optional[Callable] = None,
) -> tuple:
    """
    This function converts a phase array or tuple iterable, into the corresponding mapping of their bitstring required within a particular bit-phase mapping. A ``phase_array`` iterable is provided, and each phase is mapped to a particular bitstring based on the ``phase_bit_dataframe``. A tuple is composed of strings that represent the bitstrings of the phases provided.

    Args:
        phase_array(Iterable): Iterable of phases to map to bitstrings.
        phase_bit_dataframe(pd.DataFrame): Dataframe containing the phase-bit mapping.
        phase_series_name(str): Name of the phase series in the dataframe.
        bit_series_name(str): Name of the bit series in the dataframe.
        rounding_function(Callable): Rounding function to apply to the target phase.

    Returns:
        bit_array(tuple): Tuple of bitstrings corresponding to the phases.
    """
    bit_array = []

    for phase in phase_array:
        # Apply rounding function if provided
        if rounding_function:
            phase = rounding_function(phase)

        # Check if phase is in the dataframe
        matched_rows = phase_bit_dataframe.loc[
            phase_bit_dataframe[phase_series_name] == phase, bit_series_name
        ]

        # If exact phase is not found, use the nearest phase bit representation
        if matched_rows.empty:
            bitstring, _ = find_nearest_bit_for_phase(
                phase,
                phase_bit_dataframe,
                phase_series_name,
                bit_series_name,
                rounding_function,
            )
        else:
            bitstring = matched_rows.iloc[0]

        bit_array.append(bitstring)

    return tuple(bit_array)


def extract_phase(phase_transition_list: list[FockStatePhaseTransitionType],
                  transition_type='cross'):
    """
    Extracts the phase corresponding to the specified transition type.

    Parameters:
        phase_transition_list (list of dict): Data structure containing phase transition information.
        transition_type (str): Type of transition to extract phase for ('cross' or 'bar').

    Returns:
        float: Phase corresponding to the specified transition type.
    """
    transition_mapping = {
        'cross': ((1, 0), (0, 1)),
        'bar': ((1, 0), (1, 0))
    }

    if transition_type not in transition_mapping:
        raise ValueError("Invalid transition type. Use 'cross' or 'bar'.")

    input_state, output_state = transition_mapping[transition_type]

    for entry in phase_transition_list:
        if entry['input_fock_state'] == input_state and entry['output_fock_state'] == output_state:
            return entry['phase'][0]

    raise ValueError(f"Phase for the {transition_type} transition not found.")


def find_nearest_bit_for_phase(
    target_phase: float,
    phase_bit_dataframe: pd.DataFrame,
    phase_series_name: str = "phase",
    bit_series_name: str = "bit",
    rounding_function: Optional[Callable] = None,
) -> tuple:
    """
    This is a mapping function between a provided target phase that might be more analogous, with the closest
    bit-value in a `bit-phase` ideal relationship. The error between the target phase and the applied phase is
    limited to the discretisation error of the phase mapping.

    Args:
        target_phase(float): Target phase to map to.
        phase_bit_dataframe(pd.DataFrame): Dataframe containing the phase-bit mapping.
        phase_series_name(str): Name of the phase series in the dataframe.
        bit_series_name(str): Name of the bit series in the dataframe.
        rounding_function(Callable): Rounding function to apply to the target phase.

    Returns:
        bitstring(str): Bitstring corresponding to the nearest phase.
    """
    # Apply rounding function if provided
    if rounding_function:
        target_phase = rounding_function(target_phase)

    # Find the nearest phase from the dataframe
    phases = phase_bit_dataframe[phase_series_name].values
    nearest_phase = phases[
        np.argmin(np.abs(phases - target_phase))
    ]  # TODO implement rounding function here.

    # Get the corresponding bitstring for the nearest phase
    bitstring = phase_bit_dataframe.loc[
        phase_bit_dataframe[phase_series_name] == nearest_phase, bit_series_name
    ].iloc[0]

    return bitstring, nearest_phase


def format_electro_optic_fock_transition(
    switch_state_array: array_types,
    input_fock_state_array: array_types,
    raw_output_state: array_types,
) -> FockStatePhaseTransitionType:
    """
    Formats the electro-optic state into a standard FockStatePhaseTransitionType format. This is useful for the
    electro-optic model to ensure that the output state is in the correct format. The output state is a dictionary
    that contains the phase, input fock state, and output fock state. The idea is that this will allow us to
    standardise and compare the output states of the electro-optic model across multiple formats.

    Args:
        switch_state_array(array_types): Array of switch states.
        input_fock_state_array(array_types): Array of valid input fock states.
        raw_output_state(array_types): Array of raw output state.

    Returns:
        electro_optic_state(FockStatePhaseTransitionType): Electro-optic state.
    """
    electro_optic_state = {
        "phase": convert_array_type(switch_state_array, "tuple"),
        "input_fock_state": convert_array_type(input_fock_state_array, tuple_int_type),
        "output_fock_state": absolute_to_threshold(raw_output_state, output_array_type=tuple_int_type),
    }
    # assert type(electro_optic_state) == FockStatePhaseTransitionType # TODO fix this
    return electro_optic_state


def get_state_phase_transitions(
    switch_function: Callable,
    switch_states: list[NumericalTypes] | None = None,
    input_fock_states: list[ArrayTypes] | None = None,
    mode_amount: int | None = None,
    **kwargs
) -> list[ArrayTypes]:
    """
    The goal of this function is to extract the corresponding phase required to implement a state transition.

    Let's consider a simple MZI 2x2 logic with two transmission states. We want to verify that the electronic function
    switch, effectively switches the optical output between the cross and bar states of the optical transmission function.

    For the corresponding switch model:

    Let's assume a switch model unitary. For a given 2x2 input optical switch "X". In bar state, in dual rail, transforms an optical input:
    ```
    .. raw::

        [[1] ----> [[1]
        [0]]        [0]]

    In cross state, in dual rail, transforms an optical input:

    .. raw::

        [[1] ----> [[0]
        [0]]        [1]]

    However, sometimes it is easier to describe a photonic logic transformation based on these states, rather than inherently
    the numerical phase that is applied. This may be the case, for example, in asymmetric Mach-Zehnder modulators models, etc.

    As such, this function will help us extract the corresponding phase for a particular switch transition.
    """
    # We compose the switch_states we want to apply
    if switch_states is None:
        switch_states = [0, jnp.pi]

    # We compose the fock states we want to apply
    if input_fock_states is None:
        input_fock_states = fock_states_only_individual_modes(
            mode_amount=mode_amount,
            maximum_photon_amount=1,
            output_type="jax",
        )

    circuits = list()
    output_states = list()
    for switch_state_i in switch_states:
        # Get the transmission matrix for the switch state
        circuit_i = sax_to_s_parameters_standard_matrix(
            # TODO maybe generalise the switch address state mapping into a corresponding function
            switch_function(
                sxt={"active_phase_rad": switch_state_i}
            ),
            **kwargs
        )

        # See if the switch state is correctly applied to the input fock states
        for input_fock_state_i in input_fock_states:
            raw_output_state_i = jnp.dot(circuit_i[0], input_fock_state_i)
            output_state_i = format_electro_optic_fock_transition(
                switch_state_array=(switch_state_i,),
                input_fock_state_array=input_fock_state_i,
                raw_output_state=raw_output_state_i
            )
            output_states.append(output_state_i)
            # Now we need to find a way to verify that the model is correct by comparing to our expectation output.
            # We can do this by comparing the output state to the target fock state.
    return output_states


def get_state_to_phase_map(
    switch_function: Callable,
    switch_states: list[NumericalTypes] | None = None,
    input_fock_states: list[ArrayTypes] | None = None,
    target_transition_list: list[dict] | None = None,
    mode_amount: int | None = None,
    **kwargs
) -> tuple[ArrayTypes]:
    """
    The goal of this function is to extract the corresponding phase required to implement a state transition.

    Let's consider a simple MZI 2x2 logic with two transmission states. We want to verify that the electronic function
    switch, effectively switches the optical output between the cross and bar states of the optical transmission function.

    For the corresponding switch model:

    Let's assume a switch model unitary. For a given 2x2 input optical switch "X". In bar state, in dual rail, transforms an optical input:
    ```
    .. raw::

        [[1] ----> [[1]
        [0]]        [0]]

    In cross state, in dual rail, transforms an optical input:

    .. raw::

        [[1] ----> [[0]
        [0]]        [1]]

    However, sometimes it is easier to describe a photonic logic transformation based on these states, rather than inherently
    the numerical phase that is applied. This may be the case, for example, in asymmetric Mach-Zehnder modulators models, etc.

    As such, this function will help us extract the corresponding phase for a particular switch transition.
    """
    state_phase_transition_list = get_state_phase_transitions(switch_function=switch_function, switch_states=switch_states,
                                                              input_fock_states=input_fock_states, mode_amount=mode_amount, **kwargs)
    # TODO implement the extraction from mapping the target fock states to the corresponing phase in more generic way
    cross_phase = extract_phase(state_phase_transition_list, transition_type='cross')
    bar_phase = extract_phase(state_phase_transition_list, transition_type='bar')
    return bar_phase, cross_phase


def linear_bit_phase_map(
    bits_amount: int,
    final_phase_rad: float,
    initial_phase_rad: float = 0,
    return_dataframe: bool = True,
    quantization_error: float = 0.000001,
    bit_format: Literal["int", "str"] = "int",
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

    if bit_format == "int":
        pass
    elif bit_format == "str":
        bits_array = bits_array_from_bits_amount(bits_amount, bit_format=bit_format)

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
