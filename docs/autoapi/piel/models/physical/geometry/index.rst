:py:mod:`piel.models.physical.geometry`
=======================================

.. py:module:: piel.models.physical.geometry


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.geometry.calculate_cross_sectional_area_m2
   piel.models.physical.geometry.awg_to_cross_sectional_area_m2



.. py:function:: calculate_cross_sectional_area_m2(diameter_m: float) -> float

   Calculates the cross-sectional area of a circle in meters squared.

   :param diameter_m: Diameter of the circle in meters.
   :type diameter_m: float

   :returns: Cross sectional area in meters squared.
   :rtype: float


.. py:function:: awg_to_cross_sectional_area_m2(awg: int) -> float

   Converts an AWG value to the cross-sectional area in meters squared.

   :param awg: The AWG value to convert.
   :type awg: int

   :returns: The cross-sectional area in meters squared.
   :rtype: float
