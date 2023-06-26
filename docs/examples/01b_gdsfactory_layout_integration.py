# # GDSFactory-OpenLane Layout Integration

# This is the simplest implementation of the integration between `OpenLane` and `gdsfactory`.
import piel

# We can find which is the latest design run

latest_design_run_directory = piel.find_design_run(
    design_directory="inverter",
)
latest_design_run_directory


# We can test this with a provided pre-build OpenLane design.

inverter_component = piel.create_gdsfactory_component_from_openlane(
    design_directory="inverter",
)

inverter_component
