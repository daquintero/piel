:py:mod:`piel.models.physical.electrical.cable`
===============================================

.. py:module:: piel.models.physical.electrical.cable


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electrical.cable.calculate_coaxial_cable_geometry
   piel.models.physical.electrical.cable.define_coaxial_cable_materials
   piel.models.physical.electrical.cable.calculate_coaxial_cable_heat_transfer
   piel.models.physical.electrical.cable.calculate_dc_cable_geometry
   piel.models.physical.electrical.cable.define_dc_cable_materials
   piel.models.physical.electrical.cable.calculate_dc_cable_heat_transfer



.. py:function:: calculate_coaxial_cable_geometry(length_m: float = 1, sheath_top_diameter_m: float = 0.001651, sheath_bottom_diameter_m: float = 0.001468, core_diameter_dimension: Literal[awg, metric] = 'metric', core_diameter_awg: piel.models.physical.electrical.types.Optional[float] = None, core_diameter_m: float = 0.002, **kwargs) -> piel.models.physical.electrical.types.CoaxialCableGeometryType

   Calculate the geometry of a coaxial cable. Defaults are based on the parameters of a TODO

   :param length_m: Length of the cable in meters.
   :param sheath_top_diameter_m: Diameter of the top of the sheath in meters.
   :param sheath_bottom_diameter_m: Diameter of the bottom of the sheath in meters.
   :param core_diameter_dimension: Dimension of the core diameter.
   :param core_diameter_awg: Core diameter in AWG.
   :param core_diameter_m: Core diameter in meters.
   :param \*\*kwargs:

   :returns: The geometry of the coaxial cable.
   :rtype: CoaxialCableGeometryType


.. py:function:: define_coaxial_cable_materials(core_material: piel.models.physical.electrical.types.MaterialReferenceType, sheath_material: piel.models.physical.electrical.types.MaterialReferenceType, dielectric_material: piel.models.physical.electrical.types.MaterialReferenceType) -> piel.models.physical.electrical.types.CoaxialCableMaterialSpecificationType

   Define the materials of a coaxial cable.

   :param core_material: The material of the core.
   :param sheath_material: The material of the sheath.
   :param dielectric_material: The material of the dielectric.

   :returns: The material specification of the coaxial cable.
   :rtype: CoaxialCableMaterialSpecificationType


.. py:function:: calculate_coaxial_cable_heat_transfer(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, geometry_class: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.CoaxialCableGeometryType], material_class: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.CoaxialCableMaterialSpecificationType], core_material: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.MaterialReferenceType] = None, sheath_material: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.MaterialReferenceType] = None, dielectric_material: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.MaterialReferenceType] = None) -> piel.models.physical.electrical.types.CoaxialCableHeatTransferType

   Calculate the heat transfer of a coaxial cable.

   :param temperature_range_K: The temperature range in Kelvin.
   :param geometry_class: The geometry of the cable.
   :param material_class: The material of the cable.
   :param core_material: The material of the core.
   :param sheath_material: The material of the sheath.
   :param dielectric_material: The material of the dielectric.

   :returns: The heat transfer of the cable.
   :rtype: CoaxialCableHeatTransferType


.. py:function:: calculate_dc_cable_geometry(length_m: float = 1, core_diameter_dimension: Literal[awg, metric] = 'metric', core_diameter_awg: piel.models.physical.electrical.types.Optional[float] = None, core_diameter_m: float = 0.002, *args, **kwargs) -> piel.models.physical.electrical.types.DCCableGeometryType

   Calculate the geometry of a DC cable. Defaults are based on the parameters of a TODO

   :param length_m: Length of the cable in meters.
   :param core_diameter_dimension: Dimension of the core diameter.
   :param core_diameter_awg: Core diameter in AWG.
   :param core_diameter_m: Core diameter in meters.
   :param \*\*kwargs:

   :returns: The geometry of the coaxial cable.
   :rtype: CoaxialCableGeometryType


.. py:function:: define_dc_cable_materials(core_material: piel.models.physical.electrical.types.MaterialReferenceType) -> piel.models.physical.electrical.types.DCCableMaterialSpecificationType

   Define the materials of a coaxial cable.

   :param core_material: The material of the core.

   :returns: The material specification of the dc cable.
   :rtype: DCCableMaterialSpecificationType


.. py:function:: calculate_dc_cable_heat_transfer(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, geometry_class: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.DCCableGeometryType], material_class: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.DCCableMaterialSpecificationType], core_material: piel.models.physical.electrical.types.Optional[piel.models.physical.electrical.types.MaterialReferenceType] = None) -> piel.models.physical.electrical.types.DCCableHeatTransferType

   Calculate the heat transfer of a coaxial cable.

   :param temperature_range_K: The temperature range in Kelvin.
   :param geometry_class: The geometry of the cable.
   :param material_class: The material of the cable.
   :param core_material: The material of the core.

   :returns: The heat transfer of the cable.
   :rtype: DCCableHeatTransferType
