# # Electronic-Photonic Timing Networks

# One of the main complexities of creating concurrent photonic-electronics systems is matching the timing between the optical signals and the electronic signals in relation to a datum reference. These concucrrent systems can hard enough to conceptualize, and hence `piel` provides some functionality to enable easier network analysis.

# ## The Basics

# ### `Connection` with `time`

# Part of the desired functionality in `piel` is to be able to understand these timing networks in relation to existing component definitions. Each `Connection` has a `time` definition which is defined as `TimeMetrics`.

# Let's start by creating a network of two paths.

import piel
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Let's create a basic network between two paths.

port_1 = piel.types.Port()
port_2 = piel.types.Port()
connection = piel.types.Connection(ports=[port_1, port_2])
connection

# ```
# Connection(name='', attrs={}, ports=[Port(name='', attrs={}, parent_component_name=''), Port(name='', attrs={}, parent_component_name='')], time=TimeMetrics(name='', attrs={}, value=0, mean=0, min=0, max=0, standard_deviation=0))
# ```

# You will note that this `Connection` has some timing information attached to it. We can compose relevant timing information accordingly in the `Connection` definition:

basic_timing = piel.types.TimeMetrics(value=1)
timed_connection = piel.types.Connection(ports=[port_1, port_2], time=basic_timing)
timed_connection

# Using this functionality, now we can compute the total timing path of a given directional path. Now this is useful if we can define a `Component` and create connectivity accordingly.

# ### Modelling RF & Photonics Propagation
#
# A few important terms to understand are group velocity and group delay in a RF or photonic network. Basically, an RF or optical pulse is a collection of frequencies each which propagate slightly differently through a dispersive material such as many dielectrics which are used in coaxial cables or waveguides. This has a strong relationship to path-length matching in concurrent electronic-photonic systems.
#
# Let's create a cable and assign a corresponding group delay to it based on a given function:

from piel.models.physical.electrical.cables.rf import create_coaxial_cable

example_coaxial_cable = create_coaxial_cable(name="example_cable")
example_coaxial_cable

# ```python
# CoaxialCable(name='example_cable', attrs={}, ports=[PhysicalPort(name='in', attrs={}, parent_component_name='', domain=None, connector='', manifold=''), PhysicalPort(name='out', attrs={}, parent_component_name='', domain=None, connector='', manifold='')], connections=[PhysicalConnection(connections=[Connection(name='', attrs={}, ports=[PhysicalPort(name='in', attrs={}, parent_component_name='', domain=None, connector='', manifold=''), PhysicalPort(name='out', attrs={}, parent_component_name='', domain=None, connector='', manifold='')], time=TimeMetrics(name='', attrs={}, value=0, mean=0, min=0, max=0, standard_deviation=0, unit=Unit(name='second', datum='second', base=1)))], components=[])], components=[], environment=Environment(temperature_K=None, region=None), manufacturer='', model='', network=None, metrics=FrequencyTransmissionMetricsCollection(bandwidth_Hz=ScalarMetrics(name='', attrs={}, value=None, mean=None, min=None, max=None, standard_deviation=None, unit=Unit(name='ratio', datum='1', base=1)), center_transmission_dB=ScalarMetrics(name='', attrs={}, value=None, mean=None, min=None, max=None, standard_deviation=None, unit=Unit(name='ratio', datum='1', base=1))), geometry=CoaxialCableGeometryType(units=None, core_cross_sectional_area_m2=3.141592653589793e-06, length_m=1.0, sheath_cross_sectional_area_m2=4.482872075095052e-07, total_cross_sectional_area_m2=3.5898798610992983e-06), heat_transfer=CoaxialCableHeatTransferType(units='W', core=0, sheath=0, dielectric=0, total=0.0), material_specification=None)
# ```

example_sequential_component = piel.create_sequential_component_path(
    components=[example_coaxial_cable, example_coaxial_cable]
)
example_sequential_component
