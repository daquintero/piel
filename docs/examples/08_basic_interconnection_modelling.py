# # Basic Interconnection Modelling
#
# It is very difficult to design an electronic-photonic system without actually *connecting* them together. As it turns out, interconnection modelling is crucial in understanding the scaling of these systems. 
#
# We might want to model:
#
# - Interconnection effects of electro-optic switches with their transmission line, or transmission lines towards RF analog modulation
# - Path-length matched digital transmission lines to an amplifier
# - The transmission line from a detector diode to the low-noise amplifier before an analogue-to-digital converter.
# - Electronic to photonic chip bonding.
#
# An important aspect to note is that the computational architecture is designed to have the minimal amouunt of operations being computed for a given type of modelling, and this means the modelling speed is as fast as can be for a given operation within a python implementation, pending further toolset implementations.
#
# As such, understanding interconnection effects turns out to be pretty important in these type of systems.

import numpy as np
import piel
from piel.models.physical.electrical.cable import calculate_coaxial_cable_geometry, calculate_coaxial_cable_heat_transfer, calculate_dc_cable_geometry
from piel.models.physical.electrical.types import CoaxialCableGeometryType, CoaxialCableMaterialSpecificationType

# ## Starting from the Basics

# ### Modelling a DC Wire
#
# Let's take the most basic example to physically verify that the numerical functional implementation gives accurate results in terms of calculating the corresponding heat transfer. We will also do some analytical comparisons:

basic_dc_cable = calculate_dc_cable_geometry(
    length_m = 1,
    core_diameter_m = 1e-3,
)
basic_dc_cable

# ### Modelling a Coaxial Cable
#
# #### Thermal Heat Transfer
#
# Note that we have strongly-typed classes in order to manage the data containers across multiple functions. This enables flexibly extending the corresponding implementations.

basic_coaxial_cable = calculate_coaxial_cable_geometry(
    length_m = 1,
    sheath_top_diameter_m = 1.651e-3,
    sheath_bottom_diameter_m = 1.468e-3,
    core_diameter_m = 2e-3,
)
basic_coaxial_cable

# ```
# CoaxialCableGeometryType(core_cross_sectional_area_m2=3.141592653589793e-06, length_m=1.0, sheath_cross_sectional_area_m2=4.482872075095052e-07, total_cross_sectional_area_m2=3.5898798610992983e-06)
# ```

# You can also run the help function to learn more of the corresponding class, on top of the existing documentation.

# +
# help(CoaxialCableGeometryType)
# -

# Now, let's apply each section with materials. First, let's work out what are all the current supported materials specifications (feel free to contribute!). Note that this is always specific to the corresponding property.

from piel.materials.thermal_conductivity import material_references as thermal_conductivity_material_references

thermal_conductivity_material_references

# ```
# [('stainless_steel', '304'),
#  ('stainless_steel', '310'),
#  ('stainless_steel', '316'),
#  ('aluminum', '1100'),
#  ('copper', 'rrr50'),
#  ('copper', 'rrr100'),
#  ('copper', 'rrr150'),
#  ('copper', 'rrr300'),
#  ('copper', 'rrr500'),
#  ('teflon', None)]
# ```

# It is pretty straightforward to define a corresponding coaxial-cable material specification accordingly with the static `CoaxialCableMaterialSpecificationType` container:

basic_coaxial_cable_materials = CoaxialCableMaterialSpecificationType(
    core=('copper', 'rrr50'),
    sheath=('copper', 'rrr50'),
    dielectric=('teflon', None)
)
basic_coaxial_cable_materials

# ```
# CoaxialCableMaterialSpecificationType(core=('copper', 'rrr50'), sheath=('copper', 'rrr50'), dielectric=('teflon', None))
# ```

# Now, let's assume we have a coaxial cable that goes from room temperature to cryogenic temperatures. Say, a cable inside a cryostat that goes from 273K to 70K. Let's work out how much thermal heat transfer occurs in between these stages in Watts.

temperature_range_limits_K = tuple([70, 273])

basic_coaxial_cable_heat_transfer = calculate_coaxial_cable_heat_transfer(
    temperature_range_K=temperature_range_limits_K,
    geometry_class=basic_coaxial_cable,
    material_class=basic_coaxial_cable_materials,
)
basic_coaxial_cable_heat_transfer

# ```
# CoaxialCableHeatTransferType(core=0.0019091610845816964, sheath=0.0019091610845816964, dielectric=0.00018867678408072714, total=0.00400699895324412)
# ```

# ### Larger System Modelling

# Modelling the heat transfer of multiple coaxial cables in parallel in a system involves basic python operations:s

parallel_cables_amount = 4
4 * basic_coaxial_cable_heat_transfer.total

# ```
# 0.01602799581297648
# ```

# Modelling the heat transfer of 4 cables in series, just involves making the cable in series:

# +
basic_coaxial_cable_4_in_series = calculate_coaxial_cable_geometry(
    length_m = 1 * 4,
    sheath_top_diameter_m = 1.651e-3,
    sheath_bottom_diameter_m = 1.468e-3,
    core_diameter_m = 2e-3,
)

basic_coaxial_cable_heat_transfer_4_in_series = calculate_coaxial_cable_heat_transfer(
    temperature_range_K=temperature_range_limits_K,
    geometry_class=basic_coaxial_cable_4_in_series,
    material_class=basic_coaxial_cable_materials,
)
basic_coaxial_cable_heat_transfer_4_in_series
# -

# ```
# CoaxialCableHeatTransferType(core=0.0004772902711454241, sheath=0.0004772902711454241, dielectric=4.7169196020181784e-05, total=0.00100174973831103)
# ```


