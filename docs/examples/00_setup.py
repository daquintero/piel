# # Project Setup Example

# ## Enter Docker Environment
#
# We first enter the Docker environment by running:
# ```
# ./start_shell.sh
# # Or ./start_vnc.sh for a graphical environment
# ```
#
# Now we are in the especially configured Docker environment under `/foss/designs` and you are able to `git clone` your projects in the special recommended structure into this directory.

# ## Install `piel`
#
# If you have not installed yet, you can do:
# ```
# pip install piel
# ```
#
# Or the latest development installation as:
#
# ```
# git clone https://github.com/daquintero/piel.git
# # cd piel
# pip install -e .
# ```

# Now you can begin running this Jupyter notebook.

# ### Verify your `piel` installation

import piel

# ## Example Setup

# We will start by first setting up the design folder. You can get your own design folder, or you can use the `docs/examples/simple_design` folder as a reference for our project. In the future, you might want to have your project as a git clonable repository you can just add to a designs folder. You can also use the example OpenLanes ones.
#
# `piel` provides a convenient function so that you can automatically set this up under the `iic-osic-tools` environment.

piel.setup_example_design()

# You can check that the design has been copied appropriately by running:

# +

piel.check_example_design()
# -

# Now we have a project we can work from.


# ## Interacting with the Environment

# ### OpenLane v1 Environment

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

openlane_installed_versions = list(pathlib.Path(os.environ["OPENLANE_ROOT"]).iterdir())
openlane_installed_versions

# This will return all the `OpenLane v1` versions that have been installed. In my case it is just `2023.05`, so I will set my OpenLane root directory to be based on the latest here:

openlane_root_directory = (
    pathlib.Path(os.environ["OPENLANE_ROOT"]) / openlane_installed_versions[-1]
)
openlane_root_directory

# We can find out all the default designs in Openlane designs accordingly

list((openlane_root_directory / "designs").iterdir())


