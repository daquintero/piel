:py:mod:`piel.models.physical`
==============================

.. py:module:: piel.models.physical


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   electrical/index.rst
   electro_optic/index.rst
   electronic/index.rst
   opto_electronic/index.rst
   photonic/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   geometry/index.rst
   thermal/index.rst
   units/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.calculate_cross_sectional_area_m2
   piel.models.physical.convert_awg_to_m2



.. py:function:: calculate_cross_sectional_area_m2(diameter_m: float) -> float

   Calculates the cross sectional area of a circle in meters squared.

   :param diameter_m: Diameter of the circle in meters.
   :type diameter_m: float

   :returns: Cross sectional area in meters squared.
   :rtype: float


.. py:function:: convert_awg_to_m2(awg: int) -> float

   Converts an AWG value to meters squared.

   Reference: https://en.wikipedia.org/wiki/American_wire_gauge

   :param awg: AWG value.
   :type awg: ing

   :returns: Cross sectional area in meters squared.
   :rtype: float
