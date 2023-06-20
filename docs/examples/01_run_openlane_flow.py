# # Run OpenLane Flow

# +

import piel

# -

# Assume we are starting from the iic-osic-home design directory, all our design files are there in the format as described in the piel `sections/project_structure` documentation.

# We first enter the Docker environment by running:
# ```
# ./start_shell.sh
# # Or ./start_vnc.sh for a graphical environment
# ```
#
# Now we are in the especially configured Docker environment under `/foss/designs` and you are able to `git clone` your projects in the special recommended structure into this directory.
#
# We will go through the process of running `Openlane v1` and `Openlane v2` configured projects:

# ## OpenLane V1 Flow

# ### Interacting with the Environment

# For those who do not like shell scripting, I am afraid to tell you there is no escape when designing digital microelectronics, it is important to learn. However, `piel` provides a set of wrappers to make the design process faster, easier and more integrated into exisisting tools.
#
# <!-- TODO ADD RST documentation links -->
# We give you a list of python functions that explain methodologies of interaction with the design project environment in `docs/sections/environment/python_useful_commands` but we will review important ones now:

# You can interact with standard `OpenLane` [environment variables](https://openlane.readthedocs.io/en/latest/reference/configuration.html) through:

import os

os.environ["OPENLANE_ROOT"]

# This gives us the source directory of the OpenLane v1 installation under `foss/tools/`, but not the version directory, nor the directory of the standard `./flow.tcl` design inclusion and execution.

# We can find out what version directory has been installed through `pathlib` functionality:

import pathlib

openlane_installed_versions = pathlib.Path(os.environ["OPENLANE_ROOT"]).iterdir()
openlane_installed_versions

# This will return all the `OpenLane v1` versions that have been installed. In my case it is just `2023.05`, so I will set my OpenLane root directory to be based on the latest here:

openlane_root_directory = (
    pathlib.Path(os.environ["OPENLANE_ROOT"]) / openlane_installed_versions[-1]
)
openlane_root_directory

# ### Run Default `spm` Design using `piel`

# #### The Fast Version

# `piel` provides a set of functions for easily configuring and running a design into `Openlane v1`. For the default `spm` that already has a set up `config.json` file and project structure:

piel.run_openlane_flow(
    design_directory="/foss/designs/spm",
)

# #### The Slow Version
