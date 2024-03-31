:py:mod:`piel.materials.thermal_conductivity`
=============================================

.. py:module:: piel.materials.thermal_conductivity


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   aluminum/index.rst
   copper/index.rst
   stainless_steel/index.rst
   teflon/index.rst
   types/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.materials.thermal_conductivity.stainless_steel
   piel.materials.thermal_conductivity.aluminum
   piel.materials.thermal_conductivity.copper
   piel.materials.thermal_conductivity.teflon



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.materials.thermal_conductivity.MaterialReferenceType
   piel.materials.thermal_conductivity.MaterialReferencesTypes
   piel.materials.thermal_conductivity.SpecificationType
   piel.materials.thermal_conductivity.stainless_steel_material_references
   piel.materials.thermal_conductivity.aluminum_material_references
   piel.materials.thermal_conductivity.copper_material_references
   piel.materials.thermal_conductivity.teflon_material_references
   piel.materials.thermal_conductivity.material_references


.. py:function:: stainless_steel(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, material_reference: piel.materials.thermal_conductivity.types.MaterialReferenceType, *args, **kwargs)


.. py:function:: aluminum(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, material_reference: piel.materials.thermal_conductivity.types.MaterialReferenceType, *args, **kwargs) -> float


.. py:function:: copper(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, material_reference: piel.materials.thermal_conductivity.types.MaterialReferenceType, *args, **kwargs) -> piel.types.ArrayTypes


.. py:function:: teflon(temperature_range_K: piel.models.physical.types.TemperatureRangeTypes, *args, **kwargs)

   Trade Names for FEP resins include DuPont Teflon™, Daikin Neoflon™, Dyneon Hostaflon™, NiFlon, Sinoflon.
   Source: https://trc.nist.gov/cryogenics/materials/Teflon/Teflon_rev.htm

   :param temperature_range_K:

   Returns:



.. py:data:: MaterialReferenceType

   

.. py:data:: MaterialReferencesTypes

   

.. py:data:: SpecificationType

   

.. py:data:: stainless_steel_material_references
   :type: piel.materials.thermal_conductivity.types.MaterialReferencesTypes

   

.. py:data:: aluminum_material_references
   :type: piel.materials.thermal_conductivity.types.MaterialReferencesTypes

   

.. py:data:: copper_material_references
   :type: piel.materials.thermal_conductivity.types.MaterialReferencesTypes

   

.. py:data:: teflon_material_references
   :type: piel.materials.thermal_conductivity.types.MaterialReferencesTypes
   :value: [('teflon', None)]

   

.. py:data:: material_references

   

