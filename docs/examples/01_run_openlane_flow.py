# # Run OpenLane Flow

import piel

# Assume we are starting from the iic-osic-home design directory, all our design files are there in the format as described in the piel `sections/project_structure` documentation. You have followed the previous environment docs/examples/00_setup_environment to run the projects in this example:
#

# We will go through the process of running `Openlane v1` and `Openlane v2` configured projects:

# ## OpenLane V1 Flow

# ### Run Default `spm` Design using `piel`

# #### The Fast Version

# `piel` provides a set of functions for easily configuring and running a design into `Openlane v1`. For the default `spm` that already has a set up `config.json` file and project structure inside `$OPENLANE_ROOT/<latestversion>/designs`:


design_name = "spm"

piel.configure_and_run_design_openlane_v1(
    design_name=design_name,
)

# #### The Slow Version


# Get the latest version OpenLane v1 root directory:

root_directory = piel.get_latest_version_root_openlane_v1()
root_directory

list((root_directory / "designs").iterdir())

# Check that the design_directory provided is under $OPENLANE_ROOT/<"latestversion">/designs:

design_exists = piel.check_design_exists_openlane_v1(design_name)
design_directory = root_directory / "designs" / design_name
design_exists

# Check if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`

config_json_exists = piel.check_config_json_exists_openlane_v1(design_name)
config_json_exists

# Create a script directory, a script is written and permissions are provided for it to be executable.

piel.configure_flow_script_openlane_v1(design_directory=design_directory, design_name=design_name)

# Permit and execute the `openlane_flow.sh` script in the `scripts` directory.

openlane_flow_script_path = design_directory / "scripts" / "openlane_flow.sh"
piel.permit_script_execution(openlane_flow_script_path)
piel.run_script(openlane_flow_script_path)

# ## OpenLane V2 Flow

piel.run_openlane_flow(
    design_directory="/foss/designs/spm",
)
