:py:mod:`piel.models.physical.electro_optic.basic_heater`
=========================================================

.. py:module:: piel.models.physical.electro_optic.basic_heater


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electro_optic.basic_heater.linear_phase_mapping_relationship



.. py:function:: linear_phase_mapping_relationship(phase_power_slope: float, zero_power_phase: float)

   This function returns a function that maps the power applied to a particular heater resistor linearly. For
   example, we might start with a minimum phase mapping of (0,0) where the units are in (Watts, Phase). If we have a ridiculous arbitrary phase_power_slope of 1rad/1W, then the points in our linear mapping would be (0,0), (1,1), (2,2), (3,3), etc. This is implemented as a lambda function that takes in a power and returns a phase. The units of the power and phase are determined by the phase_power_slope and zero_power_phase. The zero_power_phase is the phase at zero power. The phase_power_slope is the slope of the linear mapping. The units of the phase_power_slope are radians/Watt. The units of the zero_power_phase are radians. The units of the power are Watts. The units of the phase are radians.

   :param phase_power_slope: The slope of the linear mapping. The units of the phase_power_slope are radians/Watt.
   :type phase_power_slope: float
   :param zero_power_phase: The phase at zero power. The units of the zero_power_phase are radians.
   :type zero_power_phase: float

   :returns: A function that maps the power applied to a particular heater resistor linearly. The units of the power and phase are determined by the phase_power_slope and zero_power_phase. The zero_power_phase is the phase at zero power. The phase_power_slope is the slope of the linear mapping. The units of the phase_power_slope are radians/Watt. The units of the zero_power_phase are radians. The units of the power are Watts. The units of the phase are radians.
   :rtype: linear_phase_mapping (function)
