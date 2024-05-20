:py:mod:`piel.models.physical.electrical.types`
===============================================

.. py:module:: piel.models.physical.electrical.types


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   piel.models.physical.electrical.types.CoaxialCableGeometryType
   piel.models.physical.electrical.types.CoaxialCableHeatTransferType
   piel.models.physical.electrical.types.CoaxialCableMaterialSpecificationType
   piel.models.physical.electrical.types.DCCableGeometryType
   piel.models.physical.electrical.types.DCCableHeatTransferType
   piel.models.physical.electrical.types.DCCableMaterialSpecificationType




Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electrical.types.CableHeatTransferTypes
   piel.models.physical.electrical.types.CableGeometryTypes
   piel.models.physical.electrical.types.CableMaterialSpecificationTypes


.. py:class:: CoaxialCableGeometryType


   Bases: :py:obj:`piel.types.QuantityType`

   The base class for all cable types.

   .. py:attribute:: core_cross_sectional_area_m2
      :type: Optional[float]

      The cross-sectional area of the core in meters squared.

   .. py:attribute:: length_m
      :type: float

      The length of the cable in meters.

   .. py:attribute:: sheath_cross_sectional_area_m2
      :type: Optional[float]

      The cross-sectional area of the sheath in meters squared.

   .. py:attribute:: total_cross_sectional_area_m2
      :type: Optional[float]

      The total cross-sectional area of the cable in meters squared.


.. py:class:: CoaxialCableHeatTransferType


   Bases: :py:obj:`piel.types.QuantityType`

   All units are in watts.

   .. py:attribute:: core
      :type: Optional[float]

      The computed heat transfer in watts for the core of the cable.

   .. py:attribute:: sheath
      :type: Optional[float]

      The computed heat transfer in watts for the sheath of the cable.

   .. py:attribute:: dielectric
      :type: Optional[float]

      The computed heat transfer in watts for the dielectric of the cable.

   .. py:attribute:: total
      :type: float

      The total computed heat transfer in watts for the cable.


.. py:class:: CoaxialCableMaterialSpecificationType


   Bases: :py:obj:`piel.types.QuantityType`

   The base class for all cable types.

   .. py:attribute:: core
      :type: Optional[piel.materials.thermal_conductivity.types.MaterialReferenceType]

      The material of the core.

   .. py:attribute:: sheath
      :type: Optional[piel.materials.thermal_conductivity.types.MaterialReferenceType]

      The material of the sheath.

   .. py:attribute:: dielectric
      :type: Optional[piel.materials.thermal_conductivity.types.MaterialReferenceType]

      The material of the dielectric.


.. py:class:: DCCableGeometryType


   Bases: :py:obj:`piel.types.QuantityType`

   The base class for all cable types.

   .. py:attribute:: core_cross_sectional_area_m2
      :type: float

      The cross-sectional area of the core in meters squared.

   .. py:attribute:: length_m
      :type: float

      The length of the cable in meters.

   .. py:attribute:: total_cross_sectional_area_m2
      :type: float

      The total cross-sectional area of the cable in meters squared.


.. py:class:: DCCableHeatTransferType


   Bases: :py:obj:`piel.types.QuantityType`

   All units are in watts.

   .. py:attribute:: core
      :type: Optional[float]

      The computed heat transfer in watts for the core of the cable.

   .. py:attribute:: total
      :type: float

      The total computed heat transfer in watts for the cable.


.. py:class:: DCCableMaterialSpecificationType


   Bases: :py:obj:`piel.types.QuantityType`

   The base class for all cable types.

   .. py:attribute:: core
      :type: Optional[piel.materials.thermal_conductivity.types.MaterialReferenceType]

      The material of the core.


.. py:data:: CableHeatTransferTypes



.. py:data:: CableGeometryTypes



.. py:data:: CableMaterialSpecificationTypes
