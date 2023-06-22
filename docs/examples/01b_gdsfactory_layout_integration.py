# # GDSFactory Layout Integration

# This is the simplest implementation of the integration between `OpenLane` and `gdsfactory`.
import piel

# We can test this with a provided pre-build OpenLane design.

inverter_component = piel.create_gdsfactory_component_from_openlane(
    design_directory="inverter",
)
