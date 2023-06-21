# # Create and Run Parametric OpenLane Designs

import piel

# ## OpenLane v1 Flow

# Let us first copy our `simple_design` into the root design directory. Note that as in most OPENLANE environments, operations are performed with root access to a certain level and in this environment you need to do this with root access.

openlane_v1_designs_directory = piel.get_latest_version_root_openlane_v1() / "designs"

example_design_name = "simple_design"

piel.copy_source_folder(
    source_directory=example_design_name,
    target_directory=openlane_v1_designs_directory,
)

# Now we check whether the simple_design has been copied.
piel.check_design_exists_openlane_v1(example_design_name)

# # This will now become our base design.
base_design_directory = piel.get_design_directory_from_root_openlane_v1(
    design_name=example_design_name
)
base_design_directory

# We can read in the default OpenLane v1 configuration file:
base_configuration = piel.read_configuration_openlane_v1(
    design_name=example_design_name
)
base_configuration

# ## Configure Parametric Parameter
# We will create a parametric parameter that exists within the `configuration` dictionary. One very common and useful one in

# The first step is to configure a set of parametric designs based on variations of an input parameter.
