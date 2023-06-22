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
# We will create a parametric parameter that exists within the `configuration` dictionary. One very common and useful one in `OpenLane` is one of the `yosys` `SYNTH_PARAMETERS` which changes the synthesis parameters of the system. See `docs/section/parametric` for more details. We can easily create a "multi-parameter sweep" with some built-in `piel` utilities:

example_verilog_parameter_iteration = list()
for parameter_value in [2, 4, 8, 16, 32, 64, 128]:
    example_verilog_parameter_iteration.append(
        "MY_PARAMETER_VALUE=" + str(int(parameter_value))
    )

synthesis_parameter_iteration = {
    "SYNTH_PARAMETERS": example_verilog_parameter_iteration
}

# `piel` provides some useful functions to check multi-parameter sweeps:

parametrised_configuration_list = piel.configure_parametric_designs_openlane_v1(
    design_name=example_design_name,
    parameter_sweep_dictionary=synthesis_parameter_iteration,
)

# This function allows you to check the parameter configuration before creating the parametrised directories. Once you are ready you can do:

piel.create_parametric_designs_openlane_v1()


# The first step is to configure a set of parametric designs based on variations of an input parameter.
