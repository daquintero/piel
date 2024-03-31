:py:mod:`piel.models.physical.thermal`
======================================

.. py:module:: piel.models.physical.thermal


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.thermal.heat_transfer_1d_W



.. py:function:: heat_transfer_1d_W(thermal_conductivity_fit, temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, cross_sectional_area_m2: float, length_m: float, *args, **kwargs) -> float

   Calculate the heat transfer in watts for a 1D system. The thermal conductivity is assumed to be a function of
   temperature.

   .. math::

       q = A \int_{T_1}^{T_2} k(T) dT

   :param thermal_conductivity_fit:
   :param temperature_range_K:
   :param cross_sectional_area_m2:
   :param length_m:

   :returns: The heat transfer in watts for a 1D system.
   :rtype: float


