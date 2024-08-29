# # Electronic-Photonic Timing Networks

# One of the main complexities of creating concurrent photonic-electronics systems is matching the timing between the optical signals and the electronic signals in relation to a datum reference. These concucrrent systems can hard enough to conceptualize, and hence `piel` provides some functionality to enable easier network analysis.

# ## Setting up a Basic Network

# Part of the desired functionality in `piel` is to be able to understand these timing networks in relation to existing component definitions. Each `Connection` has a `time` definition which is defined as `TimeMetrics`.

# Let's start by creating a network of two paths.

import piel

# Let's create a basic network between two paths.

port_1 = piel.types.Port()
port_2 = piel.types.Port()
connection = piel.types.Connection(ports=[port_1, port_2])
connection

# ```
# Connection(name='', attrs={}, ports=[Port(name='', attrs={}, parent_component_name=''), Port(name='', attrs={}, parent_component_name='')], time=TimeMetrics(name='', attrs={}, value=0, mean=0, min=0, max=0, standard_deviation=0))
# ```

# You will note that this `Connection` has some timing information attached to it. We can compose relevant timing information accordingly in the `Connection` definition:

timing = piel.types.TimeMetrics(value=0)
timing

dir(piel.models.frequency)
