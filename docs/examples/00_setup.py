# # Project Setup Example

# ## Within [IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS)
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

piel.__version__

piel.develop.configure_development_environment()

# ## Example Setup

# We will start by first setting up the design folder. You can get your own design folder, or you can use the `docs/examples/simple_design` folder as a reference for our project. In the future, you might want to have your project as a git clonable repository you can just add to a designs folder. You can also use the example OpenLanes ones.
#
# `piel` provides a convenient function so that you can automatically set this up under the `iic-osic-tools` environment.

piel.copy_example_design()

# You can check that the design has been copied appropriately by running:

piel.check_example_design()

# Now we have a project we can work from.


# ## Recommended Project Structure

# In `docs/sections/environment/project_strucutre.md` TODO link, the recommended project structure of your codesign projects is described. This structure has been carefully thought of in order to have the most seamless co-design experience when using all the tools. It follows standard convention that merges both the `photonic` and `electronic` design flows into a single one.
#
# Install your project as a Python package through `pip install . -e`, this makes the package accessible throughout your full filesystem, and it means `piel` can interact with your design at any location in your filesystem. After you do that, you can then run the following commands.

# `piel` provides a set of useful functions to interact with your project. You can find out your Python project directory through:

import simple_design

piel.return_path(simple_design)

# ```
# WindowsPath('c:/users/dario/documents/phd/piel/docs/examples/designs/simple_design/simple_design')
# ```

# This means that in any corresponding `piel` functionality, wherever a path is required, you can just use your project module import, and the operations will be performed on that directory.

# ### Starting from a default example

# You can copy an example `piel` project structure to play around with using the following command. In this case, we are copying the `simple_design` example from our `designs` directory.

piel.copy_example_design(
    project_source="piel",  # From `piel` project examples
    example_name="simple_design",
    target_directory="designs/",
    target_project_name="simple_copied_design",
    delete=True,  # TODO update for more robust tests
)

# ### Create an empty `piel` project

# In the future we will have a `cookiecutter` that sets up the design folder structure so that you can just get cracking on a multi-physical design. For now, we have this function:

piel.create_empty_piel_project(
    project_name="example_empty_project", parent_directory="designs/"
)

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
