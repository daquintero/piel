# # Project Setup Example
import piel

# We will start by first setting up the design folder. You can get your own design folder, or you can use the `docs/examples/simple_design` folder as a reference for our project. In the future, you might want to have your project as a git clonable repository you can just add to a designs folder. You can also use the example OpenLanes ones.

# ## Example Setup
# `piel` provides a convenient function so that you can automatically set this up under the `iic-osic-tools` environment.
piel.setup_example_design()

# You can check that the design has been copied appropiately by running:
piel.check_example_design()

# Now we have a project we can work from.
