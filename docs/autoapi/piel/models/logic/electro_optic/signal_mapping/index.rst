:py:mod:`piel.models.logic.electro_optic.signal_mapping`
========================================================

.. py:module:: piel.models.logic.electro_optic.signal_mapping

.. autoapi-nested-parse::

   TODO implement this function.
   In this function we implement different methods of mapping electronic signals to phase.

   One particular implementation of phase mapping would be:

   .. list-table:: Example Basic Phase Mapping
      :header-rows: 1

      * - Bit
        - Phase
      * - b0
        - :math:`\phi_0 \to 0`
      * - b1
        - :math:`\phi_1 \to \pi`

   We can define the two corresponding angles that this would be.

   A more complex implementation of phase mapping can be similar to a DAC mapping: a bitstring within a converter
   bit-size can map directly to a particular phase space within a particular mapping.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.logic.electro_optic.signal_mapping.bits_array_from_bits_amount
   piel.models.logic.electro_optic.signal_mapping.convert_phase_array_to_bit_array
   piel.models.logic.electro_optic.signal_mapping.find_nearest_bit_for_phase
   piel.models.logic.electro_optic.signal_mapping.linear_bit_phase_map
   piel.models.logic.electro_optic.signal_mapping.return_phase_array_from_data_series
   piel.models.logic.electro_optic.signal_mapping.format_electro_optic_fock_transition



.. py:function:: bits_array_from_bits_amount(bits_amount: int, bit_format: Literal[int, str] = 'int') -> numpy.ndarray

   Returns an array of bits from a given amount of bits.

   :param bits_amount: Amount of bits to generate.
   :type bits_amount: int

   :returns: Array of bits.
   :rtype: bit_array(np.ndarray)


.. py:function:: convert_phase_array_to_bit_array(phase_array: Iterable, phase_bit_dataframe: pandas.DataFrame, phase_series_name: str = 'phase', bit_series_name: str = 'bit', rounding_function: Optional[Callable] = None) -> tuple

   This function converts a phase array or tuple iterable, into the corresponding mapping of their bitstring required within a particular bit-phase mapping. A ``phase_array`` iterable is provided, and each phase is mapped to a particular bitstring based on the ``phase_bit_dataframe``. A tuple is composed of strings that represent the bitstrings of the phases provided.

   :param phase_array: Iterable of phases to map to bitstrings.
   :type phase_array: Iterable
   :param phase_bit_dataframe: Dataframe containing the phase-bit mapping.
   :type phase_bit_dataframe: pd.DataFrame
   :param phase_series_name: Name of the phase series in the dataframe.
   :type phase_series_name: str
   :param bit_series_name: Name of the bit series in the dataframe.
   :type bit_series_name: str
   :param rounding_function: Rounding function to apply to the target phase.
   :type rounding_function: Callable

   :returns: Tuple of bitstrings corresponding to the phases.
   :rtype: bit_array(tuple)


.. py:function:: find_nearest_bit_for_phase(target_phase: float, phase_bit_dataframe: pandas.DataFrame, phase_series_name: str = 'phase', bit_series_name: str = 'bit', rounding_function: Optional[Callable] = None) -> tuple

   This is a mapping function between a provided target phase that might be more analogous, with the closest
   bit-value in a `bit-phase` ideal relationship. The error between the target phase and the applied phase is
   limited to the discretisation error of the phase mapping.

   :param target_phase: Target phase to map to.
   :type target_phase: float
   :param phase_bit_dataframe: Dataframe containing the phase-bit mapping.
   :type phase_bit_dataframe: pd.DataFrame
   :param phase_series_name: Name of the phase series in the dataframe.
   :type phase_series_name: str
   :param bit_series_name: Name of the bit series in the dataframe.
   :type bit_series_name: str
   :param rounding_function: Rounding function to apply to the target phase.
   :type rounding_function: Callable

   :returns: Bitstring corresponding to the nearest phase.
   :rtype: bitstring(str)


.. py:function:: linear_bit_phase_map(bits_amount: int, final_phase_rad: float, initial_phase_rad: float = 0, return_dataframe: bool = True, quantization_error: float = 1e-06, bit_format: Literal[int, str] = 'int') -> dict | pandas.DataFrame

   Returns a linear direct mapping of bits to phase.

   :param bits_amount: Amount of bits to generate.
   :type bits_amount: int
   :param final_phase_rad: Final phase to map to.
   :type final_phase_rad: float
   :param initial_phase_rad: Initial phase to map to.
   :type initial_phase_rad: float

   :returns: Mapping of bits to phase.
   :rtype: bit_phase_mapping(dict)


.. py:function:: return_phase_array_from_data_series(data_series: pandas.Series, phase_map: pandas.DataFrame | pandas.Series) -> list

   Returns a list of phases from a given data series and phase map.
   # TODO optimise lookup table speed

   :param data_series: Data series to map.
   :type data_series: pd.Series
   :param phase_map: Phase map to use.
   :type phase_map: pd.DataFrame | pd.Series

   :returns: List of phases.
   :rtype: phase_array(list)


.. py:function:: format_electro_optic_fock_transition(switch_state_array: piel.integration.type_conversion.array_types, input_fock_state_array: piel.integration.type_conversion.array_types, raw_output_state: piel.integration.type_conversion.array_types) -> piel.models.logic.electro_optic.types.electro_optic_fock_state_type

   Formats the electro-optic state into a standard electro_optic_fock_state_type format. This is useful for the
   electro-optic model to ensure that the output state is in the correct format. The output state is a dictionary
   that contains the phase, input fock state, and output fock state. The idea is that this will allow us to
   standardise and compare the output states of the electro-optic model across multiple formats.

   :param switch_state_array: Array of switch states.
   :type switch_state_array: array_types
   :param input_fock_state_array: Array of valid input fock states.
   :type input_fock_state_array: array_types
   :param raw_output_state: Array of raw output state.
   :type raw_output_state: array_types

   :returns: Electro-optic state.
   :rtype: electro_optic_state(electro_optic_fock_state_type)
