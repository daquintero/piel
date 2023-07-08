:py:mod:`piel.models.logic.electro_optic.signal_mapping`
========================================================

.. py:module:: piel.models.logic.electro_optic.signal_mapping

.. autoapi-nested-parse::

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

   A more complex implementation of phase mapping can be similar to a DAC mapping: a bitstring within a converter bit-size can map directly to a particular phase space within a particular mapping.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.logic.electro_optic.signal_mapping.bits_array_from_bits_amount
   piel.models.logic.electro_optic.signal_mapping.linear_bit_phase_map
   piel.models.logic.electro_optic.signal_mapping.return_phase_array_from_data_series



.. py:function:: bits_array_from_bits_amount(bits_amount: int) -> numpy.ndarray

   Returns an array of bits from a given amount of bits.

   :param bits_amount: Amount of bits to generate.
   :type bits_amount: int

   :returns: Array of bits.
   :rtype: bit_array(np.ndarray)


.. py:function:: linear_bit_phase_map(bits_amount: int, final_phase_rad: float, initial_phase_rad: float = 0, return_dataframe: bool = True, quantization_error: float = 1e-06) -> dict | pandas.DataFrame

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
