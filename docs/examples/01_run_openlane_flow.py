# # Run OpenLane Flow

# <div style="padding: 10px; border-radius: 5px;">
# <strong>⚠️ Warning:</strong> This example requires uses packages which are locally available when cloning and installing the `stable` verision of the github source code. See example setup as follows:
# </div>

# + active=""
# !git clone https://github.com/daquintero/piel.git
# !cd piel/
# !pip install -e .[tools]
# !pip install -r requirements_notebooks.txt
# -

import piel

# Assume we are starting from the iic-osic-home design directory, all our design files are there in the format as described in the piel `sections/project_structure` documentation. You have followed the previous environment docs/examples/00_setup_environment to run the projects in this example:
#

# We will go through the process of running only `Openlane v2` configured projects (v1 is not supported/depreciated).

# Each design flow might have its own implementation strategy. Combinatorial designs follow different flows to sequential designs and this can be automated if we want to quickly go through an implementation strategy. Let's verify our implementation configuration.


# ## Basic Inverter Project

# +
# # Uncomment to configure
# # To configure the project in a `piel` compatible directory format.
# piel.create_empty_piel_project(
#     project_name="inverter",
#     parent_directory="designs/"
# )
# # Now copy the project from openlane-ci
# # Now let's pip install this
# # ! pip install -e designs/inverter
# # You might have to resetart the kernel after this
# -

import inverter

# Now let's verify that we can implement this project using the raw `openlane2` script per the `config.json` file provided.

dir(piel.file_system)

# ! openlane designs/inverter/inverter/config.json

# ## Amaranth-Driven Flow

# It might be desired to easily go from a design directory to an actual silicon chip purely from python. In this section of the example we will use the digital design files in an `amaranth`-generated design flow and use `openlane2` to perform the hardening of the logic.
#
# There is further documentation on migrating from the `Openlane v1` flow to `v2` in the following links:

# You need to make sure you have installed `amaranth_driven_flow` as part of the `02_digital_design_simulation` example instructions.

import amaranth_driven_flow
import piel

# The project directory is found here if you have installed the project python module:

piel.return_path(amaranth_driven_flow)


openlane_2_run_amaranth_flow = piel.tools.openlane.run_openlane_flow(
    design_directory=amaranth_driven_flow,
    only_generate_flow_setup=True,
)

# This should generate a `openlane 2` driven layout in the `amaranth_driven_flow` directory if you change the `only_generate_configuration` flag to `True`. Let's list the available runs in this project:

all_amaranth_driven_design_runs = piel.tools.openlane.find_all_design_runs(
    design_directory=amaranth_driven_flow,
)
all_amaranth_driven_design_runs

# ```python
# {'v2': [PosixPath('/home/daquintero/piel/docs/examples/designs/amaranth_driven_flow/amaranth_driven_flow/runs/RUN_2023-09-06_14-17-18')],
#  'v1': [PosixPath('/home/daquintero/piel/docs/examples/designs/amaranth_driven_flow/amaranth_driven_flow/runs/RUN_2023.08.22_00.06.09')]}
# ```

latest_amaranth_driven_openlane_runs = piel.tools.openlane.find_latest_design_run(
    design_directory=amaranth_driven_flow,
)
latest_amaranth_driven_openlane_runs

# We can check what is the path to our generated `gds` file accordingly:

piel.tools.openlane.get_gds_path_from_design_run(
    design_directory=amaranth_driven_flow,
)

# It is quite easy to visualise it on the jupyter lab using the `gdsfactory` integration widget:

amaranth_driven_flow_component = (
    piel.integration.create_gdsfactory_component_from_openlane(
        design_directory=amaranth_driven_flow,
    )
)
amaranth_driven_flow_component

# ![amaranth_driven_flow_klive](../_static/img/examples/01b_gdsfactory_layout_integation/amaranth_driven_flow_klive.PNG)

# Very cool! So now we can interact and generate `openlane 2` designs easily. Let's explore now getting some design metrics out of this design.

# ### Design Metrics Analysis

# `openlane 2` is a very powerful tool for digital design space exploration, and it is nice to be able to explore the design metrics in a pythonic way. You can explore most meaningful design metrics in `metrics.json` for a particular design.

piel.read_metrics_openlane_v2(amaranth_driven_flow)

# ```python
# {'design__instance__count': 4,
#  'design__instance_unmapped__count': 0,
#  'synthesis__check_error__count': 0,
#  'design__max_slew_violation__count__corner:nom_tt_025C_1v80': 0,
#  'design__max_fanout_violation__count__corner:nom_tt_025C_1v80': 0,
#  'design__max_cap_violation__count__corner:nom_tt_025C_1v80': 0,
#  'clock__skew__worst_hold__corner:nom_tt_025C_1v80': 0.0,
#  'clock__skew__worst_setup__corner:nom_tt_025C_1v80': 0.0,
#  'timing__hold__ws__corner:nom_tt_025C_1v80': 4.192324,
#  'timing__setup__ws__corner:nom_tt_025C_1v80': 5.192763,
#  'timing__hold__tns__corner:nom_tt_025C_1v80': 0.0,
#  'timing__setup__tns__corner:nom_tt_025C_1v80': 0.0,
#  'timing__hold__wns__corner:nom_tt_025C_1v80': 0.0,
#  'timing__setup__wns__corner:nom_tt_025C_1v80': 0.0,
#  'timing__hold_vio__count__corner:nom_tt_025C_1v80': 0,
#  'timing__hold_r2r_vio__count__corner:nom_tt_025C_1v80': 0,
#  'timing__setup_vio__count__corner:nom_tt_025C_1v80': 0,
#  'timing__setup_r2r_vio__count__corner:nom_tt_025C_1v80': 0,
#  'design__max_slew_violation__count': 0,
#  'design__max_fanout_violation__count': 0,
#  'design__max_cap_violation__count': 0,
#  'clock__skew__worst_hold': 0.0,
#  'clock__skew__worst_setup': 0.0,
#  'timing__hold__ws': 4.040462,
#  'timing__setup__ws': 4.594259,
#  'timing__hold__tns': 0.0,
#  'timing__setup__tns': 0.0,
#  'timing__hold__wns': 0.0,
#  'timing__setup__wns': 0.0,
#  'timing__hold_vio__count': 0,
#  'timing__hold_r2r_vio__count': 0,
#  'timing__setup_vio__count': 0,
#  'timing__setup_r2r_vio__count': 0,
#  'design__die__bbox': '0.0 0.0 34.5 57.12',
#  'design__core__bbox': '5.52 10.88 28.98 46.24',
#  'design__instance__displacement__total': 0,
#  'design__instance__displacement__mean': 0,
#  'design__instance__displacement__max': 0,
#  'route__wirelength__estimated': 122.622,
#  'design__violations': 0,
#  'design__instance__count__setup_buffer': 0,
#  'design__instance__count__hold_buffer': 0,
#  'antenna__violating__nets': 0,
#  'antenna__violating__pins': 0,
#  'antenna__count': 0,
#  'route__net': 10,
#  'route__net__special': 2,
#  'route__drc_errors__iter:1': 0,
#  'route__wirelength__iter:1': 104,
#  'route__drc_errors': 0,
#  'route__wirelength': 104,
#  'route__vias': 32,
#  'route__vias__singlecut': 32,
#  'route__vias__multicut': 0,
#  'design__disconnected_pins__count': 0,
#  'route__wirelength__max': 37.52,
#  'design__max_slew_violation__count__corner:nom_ss_100C_1v60': 0,
#  'design__max_fanout_violation__count__corner:nom_ss_100C_1v60': 0,
#  'design__max_cap_violation__count__corner:nom_ss_100C_1v60': 0,
#  'clock__skew__worst_hold__corner:nom_ss_100C_1v60': 0.0,
#  'clock__skew__worst_setup__corner:nom_ss_100C_1v60': 0.0,
#  'timing__hold__ws__corner:nom_ss_100C_1v60': 4.558208,
#  'timing__setup__ws__corner:nom_ss_100C_1v60': 4.600908,
#  'timing__hold__tns__corner:nom_ss_100C_1v60': 0.0,
#  'timing__setup__tns__corner:nom_ss_100C_1v60': 0.0,
#  'timing__hold__wns__corner:nom_ss_100C_1v60': 0.0,
#  'timing__setup__wns__corner:nom_ss_100C_1v60': 0.0,
#  'timing__hold_vio__count__corner:nom_ss_100C_1v60': 0,
#  'timing__hold_r2r_vio__count__corner:nom_ss_100C_1v60': 0,
#  'timing__setup_vio__count__corner:nom_ss_100C_1v60': 0,
#  'timing__setup_r2r_vio__count__corner:nom_ss_100C_1v60': 0,
#  'design__max_slew_violation__count__corner:nom_ff_n40C_1v95': 0,
#  'design__max_fanout_violation__count__corner:nom_ff_n40C_1v95': 0,
#  'design__max_cap_violation__count__corner:nom_ff_n40C_1v95': 0,
#  'clock__skew__worst_hold__corner:nom_ff_n40C_1v95': 0.0,
#  'clock__skew__worst_setup__corner:nom_ff_n40C_1v95': 0.0,
#  'timing__hold__ws__corner:nom_ff_n40C_1v95': 4.04327,
#  'timing__setup__ws__corner:nom_ff_n40C_1v95': 5.398018,
#  'timing__hold__tns__corner:nom_ff_n40C_1v95': 0.0,
#  'timing__setup__tns__corner:nom_ff_n40C_1v95': 0.0,
#  'timing__hold__wns__corner:nom_ff_n40C_1v95': 0.0,
#  'timing__setup__wns__corner:nom_ff_n40C_1v95': 0.0,
#  'timing__hold_vio__count__corner:nom_ff_n40C_1v95': 0,
#  'timing__hold_r2r_vio__count__corner:nom_ff_n40C_1v95': 0,
#  'timing__setup_vio__count__corner:nom_ff_n40C_1v95': 0,
#  'timing__setup_r2r_vio__count__corner:nom_ff_n40C_1v95': 0,
#  'design__max_slew_violation__count__corner:min_tt_025C_1v80': 0,
#  'design__max_fanout_violation__count__corner:min_tt_025C_1v80': 0,
#  'design__max_cap_violation__count__corner:min_tt_025C_1v80': 0,
#  'clock__skew__worst_hold__corner:min_tt_025C_1v80': 0.0,
#  'clock__skew__worst_setup__corner:min_tt_025C_1v80': 0.0,
#  'timing__hold__ws__corner:min_tt_025C_1v80': 4.188328,
#  'timing__setup__ws__corner:min_tt_025C_1v80': 5.195879,
#  'timing__hold__tns__corner:min_tt_025C_1v80': 0.0,
#  'timing__setup__tns__corner:min_tt_025C_1v80': 0.0,
#  'timing__hold__wns__corner:min_tt_025C_1v80': 0.0,
#  'timing__setup__wns__corner:min_tt_025C_1v80': 0.0,
#  'timing__hold_vio__count__corner:min_tt_025C_1v80': 0,
#  'timing__hold_r2r_vio__count__corner:min_tt_025C_1v80': 0,
#  'timing__setup_vio__count__corner:min_tt_025C_1v80': 0,
#  'timing__setup_r2r_vio__count__corner:min_tt_025C_1v80': 0,
#  'design__max_slew_violation__count__corner:min_ss_100C_1v60': 0,
#  'design__max_fanout_violation__count__corner:min_ss_100C_1v60': 0,
#  'design__max_cap_violation__count__corner:min_ss_100C_1v60': 0,
#  'clock__skew__worst_hold__corner:min_ss_100C_1v60': 0.0,
#  'clock__skew__worst_setup__corner:min_ss_100C_1v60': 0.0,
#  'timing__hold__ws__corner:min_ss_100C_1v60': 4.551795,
#  'timing__setup__ws__corner:min_ss_100C_1v60': 4.607092,
#  'timing__hold__tns__corner:min_ss_100C_1v60': 0.0,
#  'timing__setup__tns__corner:min_ss_100C_1v60': 0.0,
#  'timing__hold__wns__corner:min_ss_100C_1v60': 0.0,
#  'timing__setup__wns__corner:min_ss_100C_1v60': 0.0,
#  'timing__hold_vio__count__corner:min_ss_100C_1v60': 0,
#  'timing__hold_r2r_vio__count__corner:min_ss_100C_1v60': 0,
#  'timing__setup_vio__count__corner:min_ss_100C_1v60': 0,
#  'timing__setup_r2r_vio__count__corner:min_ss_100C_1v60': 0,
#  'design__max_slew_violation__count__corner:min_ff_n40C_1v95': 0,
#  'design__max_fanout_violation__count__corner:min_ff_n40C_1v95': 0,
#  'design__max_cap_violation__count__corner:min_ff_n40C_1v95': 0,
#  'clock__skew__worst_hold__corner:min_ff_n40C_1v95': 0.0,
#  'clock__skew__worst_setup__corner:min_ff_n40C_1v95': 0.0,
#  'timing__hold__ws__corner:min_ff_n40C_1v95': 4.040462,
#  'timing__setup__ws__corner:min_ff_n40C_1v95': 5.400592,
#  'timing__hold__tns__corner:min_ff_n40C_1v95': 0.0,
#  'timing__setup__tns__corner:min_ff_n40C_1v95': 0.0,
#  'timing__hold__wns__corner:min_ff_n40C_1v95': 0.0,
#  'timing__setup__wns__corner:min_ff_n40C_1v95': 0.0,
#  'timing__hold_vio__count__corner:min_ff_n40C_1v95': 0,
#  'timing__hold_r2r_vio__count__corner:min_ff_n40C_1v95': 0,
#  'timing__setup_vio__count__corner:min_ff_n40C_1v95': 0,
#  'timing__setup_r2r_vio__count__corner:min_ff_n40C_1v95': 0,
#  'design__max_slew_violation__count__corner:max_tt_025C_1v80': 0,
#  'design__max_fanout_violation__count__corner:max_tt_025C_1v80': 0,
#  'design__max_cap_violation__count__corner:max_tt_025C_1v80': 0,
#  'clock__skew__worst_hold__corner:max_tt_025C_1v80': 0.0,
#  'clock__skew__worst_setup__corner:max_tt_025C_1v80': 0.0,
#  'timing__hold__ws__corner:max_tt_025C_1v80': 4.196024,
#  'timing__setup__ws__corner:max_tt_025C_1v80': 5.189277,
#  'timing__hold__tns__corner:max_tt_025C_1v80': 0.0,
#  'timing__setup__tns__corner:max_tt_025C_1v80': 0.0,
#  'timing__hold__wns__corner:max_tt_025C_1v80': 0.0,
#  'timing__setup__wns__corner:max_tt_025C_1v80': 0.0,
#  'timing__hold_vio__count__corner:max_tt_025C_1v80': 0,
#  'timing__hold_r2r_vio__count__corner:max_tt_025C_1v80': 0,
#  'timing__setup_vio__count__corner:max_tt_025C_1v80': 0,
#  'timing__setup_r2r_vio__count__corner:max_tt_025C_1v80': 0,
#  'design__max_slew_violation__count__corner:max_ss_100C_1v60': 0,
#  'design__max_fanout_violation__count__corner:max_ss_100C_1v60': 0,
#  'design__max_cap_violation__count__corner:max_ss_100C_1v60': 0,
#  'clock__skew__worst_hold__corner:max_ss_100C_1v60': 0.0,
#  'clock__skew__worst_setup__corner:max_ss_100C_1v60': 0.0,
#  'timing__hold__ws__corner:max_ss_100C_1v60': 4.562827,
#  'timing__setup__ws__corner:max_ss_100C_1v60': 4.594259,
#  'timing__hold__tns__corner:max_ss_100C_1v60': 0.0,
#  'timing__setup__tns__corner:max_ss_100C_1v60': 0.0,
#  'timing__hold__wns__corner:max_ss_100C_1v60': 0.0,
#  'timing__setup__wns__corner:max_ss_100C_1v60': 0.0,
#  'timing__hold_vio__count__corner:max_ss_100C_1v60': 0,
#  'timing__hold_r2r_vio__count__corner:max_ss_100C_1v60': 0,
#  'timing__setup_vio__count__corner:max_ss_100C_1v60': 0,
#  'timing__setup_r2r_vio__count__corner:max_ss_100C_1v60': 0,
#  'design__max_slew_violation__count__corner:max_ff_n40C_1v95': 0,
#  'design__max_fanout_violation__count__corner:max_ff_n40C_1v95': 0,
#  'design__max_cap_violation__count__corner:max_ff_n40C_1v95': 0,
#  'clock__skew__worst_hold__corner:max_ff_n40C_1v95': 0.0,
#  'clock__skew__worst_setup__corner:max_ff_n40C_1v95': 0.0,
#  'timing__hold__ws__corner:max_ff_n40C_1v95': 4.045971,
#  'timing__setup__ws__corner:max_ff_n40C_1v95': 5.395393,
#  'timing__hold__tns__corner:max_ff_n40C_1v95': 0.0,
#  'timing__setup__tns__corner:max_ff_n40C_1v95': 0.0,
#  'timing__hold__wns__corner:max_ff_n40C_1v95': 0.0,
#  'timing__setup__wns__corner:max_ff_n40C_1v95': 0.0,
#  'timing__hold_vio__count__corner:max_ff_n40C_1v95': 0,
#  'timing__hold_r2r_vio__count__corner:max_ff_n40C_1v95': 0,
#  'timing__setup_vio__count__corner:max_ff_n40C_1v95': 0,
#  'timing__setup_r2r_vio__count__corner:max_ff_n40C_1v95': 0,
#  'design_powergrid__voltage__worst__net:VPWR__corner:nom_tt_025C_1v80': 1.8,
#  'design_powergrid__drop__average__net:VPWR__corner:nom_tt_025C_1v80': 4.69729e-11,
#  'design_powergrid__drop__worst__net:VPWR__corner:nom_tt_025C_1v80': 1.9324e-10,
#  'ir__voltage__worst': 1.8,
#  'ir__drop__avg': 4.7e-11,
#  'ir__drop__worst': 1.93e-10,
#  'design__xor_difference__count': 0,
#  'magic__drc_error__count': 0,
#  'magic__illegal_overlap__count': 0,
#  'design__lvs_device_difference__count': 0,
#  'design__lvs_net_differences__count': 0,
#  'design__lvs_property_fails__count': 0,
#  'design__lvs_errors__count': 0,
#  'design__lvs_unmatched_devices__count': 0,
#  'design__lvs_unmatched_nets__count': 0,
#  'design__lvs_unmatched_pins__count': 0}
# ```

# Some important metrics tend to be:
#
# -   `design__die__bbox` Die core size
# -   `design__core__bbox` Logic core size
# -   `design__instance__count` Amount of instances in implemented logic
#
# You might care of a few more depending on what you are aiming to do.
