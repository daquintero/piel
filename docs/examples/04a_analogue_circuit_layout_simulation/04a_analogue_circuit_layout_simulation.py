# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analogue Circuit Layout & Simulation

# ## Going through the Basics

# We will start off by importing the layout elements of the SKY130nm `gdsfactory` PDK process and understanding how we interact and map them to a schematic.

# TODO MOVE TO DOCUMENTATION WHEN MORE ADVANCED.
#
# One thing that many people might want to do is design analog circuits on chip using open-source tools.`gdsfactory` is an excellent tool for chip  layout. `hdl21` and `vlsir` are great projects to perform simulation in most open-source and proprietary SPICE simulation platforms. It makes sense to integrate the two together. We will demonstrate how to do this in this example.
#
# A collection of Google-released open-source PDKs is available [here](https://foss-eda-tools.googlesource.com/)
#
# There are several open-source PDKs. At the moment some of the most popular ones are:
#
# **SKY130nm**
# -  [Main Open-Source Release](https://github.com/google/skywater-pdk-libs-sky130_fd_sc_hd/tree/ac7fb61f06e6470b94e8afdf7c25268f62fbd7b1)
# -  [`gdsfactory` layout PDK integration](https://github.com/gdsfactory/skywater130)
# -  [`hdl21` integration](https://github.com/dan-fritchman/Hdl21/tree/main/pdks/Sky130)
#
# **GF180nm**
# -  [Main Open-Source Release](https://github.com/google/gf180mcu-pdk)
#
# **IHP130nm**
# -  [Main Open-Source Release](https://github.com/IHP-GmbH/IHP-Open-PDK)
#
# **Any PDK**
# -  [`gdsfactory` import instructions](https://gdsfactory.github.io/gdsfactory/notebooks/09_pdk_import.html)
# -  [`hdl21` instructions](https://github.com/dan-fritchman/Hdl21/tree/main/pdks/PdkTemplate)
#
# So, as we can see, there currently are many separate relevant projects that solve different aspects of an integration flow. What we would like to do is to have an integrated flow where we could, for example, do some schematic-driven-layout or some layout-driven-schematic extraction. These are some of the objectives of the flow that we would like to demonstrate.

# ## Using the `gdsfactory-sky130nm` Repository

# The goal of this notebook is to show that analogue simulation and layout can be performed together within a `gdsfactory` environment.

# +
import hdl21 as h
import gdsfactory as gf
import gplugins.hdl21 as gph
import sky130
import sky130_hdl21

from bokeh.io import output_notebook
from gdsfactory.config import rich_output
from gplugins.schematic_editor import SchematicEditor

gf.config.rich_output()

# %env BOKEH_ALLOW_WS_ORIGIN=*

output_notebook()


# +
# sky130.cells
# -

@h.module
class SkyInv:
    """ An inverter, demonstrating using PDK modules """

    # Create some IO
    i, o, VDD, VSS = h.Ports(4)

    p = sky130_hdl21.Sky130MosParams(w=1,l=1)

    # And create some transistors!
    ps = sky130_hdl21.primitives.PMOS_1p8V_STD(p)(d=o, g=i, s=VDD, b=VDD)
    ns = sky130_hdl21.primitives.NMOS_1p8V_STD(p)(d=VSS, g=i, s=o, b=VSS)


# ## Schematic-Driven-Layout

# A common analogue design flow is called schematic-driven-layout. What this entails, fundamentally, is that we design a circuit through a schematic, and then use that schematic to instruct, extract, constrain, and/or verify our circuit chip layout. This flow uses layout elements that are connected or tied to schematic symbols, and unique names that allow for identification and connectivity relationship.
#
# -  You can read [how this is done in Cadence](https://web.njit.edu/~tyson/cadence%20Layout_Tutorial.pdf)
#
# In an open-source flow, this could be, for example, demonstrated by creating a circuit using the `hdl21 schematic` tools. Each symbol would reference a specific `PCell` in the PDK. Now, we would use this individual element cell name to connect and extract to the `SPICE` model and also to the `layout` GDS cell. This allows us to connect to the separate tools for simulation and layout.
#
# Say, we can then extract a netlist from the schematic with individual cell names and PDK cells identifiers. We could in `gdsfactory` map these PDK cell identifiers to instantiate the elements in a layout. We can then use this instantated cells to perform some automatic or

# ### Manually editing the `SPICE`-generated `gdsfactory` component YAML
#
# It is important to know that with the SPICE-generated YAML, we cannot actually create a layout on its own. This is because the SPICE models do not exactly directly map to layout instances. SPICE models can represent performance corners for the same device, with multiple temperature or yield quality variations. As such, we need to assign the corresponding gds we want to layout for our specific schematic model.

example_inverter_manual_yaml = gph.generate_raw_yaml_from_module(
    SkyInv
)
print(example_inverter_manual_yaml)

# +


example_inverter_manual_yaml = """
connections:
  ns,d: ps,d
  ns,g: ps,g
  ps,d: ns,d
  ps,g: ns,g
instances:
  ns:
    component: sky130_fd_pr__nfet_01v8
    info: {}
    settings:
      As: int((nf+2)/2) * w/nf * 0.29
      ad: int((nf+1)/2) * w/nf * 0.29
      l: UNIT_1
      m: UNIT_1
      mult: UNIT_1
      nf: UNIT_1
      nrd: 0.29 / w
      nrs: 0.29 / w
      pd: 2*int((nf+1)/2) * (w/nf + 0.29)
      ports:
        b: VSS
        d: o
        g: i
        s: VSS
      ps: 2*int((nf+2)/2) * (w/nf + 0.29)
      sa: UNIT_0
      sb: UNIT_0
      sd: UNIT_0
      w: UNIT_1
  ps:
    component: sky130_fd_pr__pfet_01v8
    info: {}
    settings:
      As: int((nf+2)/2) * w/nf * 0.29
      ad: int((nf+1)/2) * w/nf * 0.29
      l: UNIT_1
      m: UNIT_1
      mult: UNIT_1
      nf: UNIT_1
      nrd: 0.29 / w
      nrs: 0.29 / w
      pd: 2*int((nf+1)/2) * (w/nf + 0.29)
      ports:
        b: VDD
        d: o
        g: i
        s: VDD
      ps: 2*int((nf+2)/2) * (w/nf + 0.29)
      sa: UNIT_0
      sb: UNIT_0
      sd: UNIT_0
      w: UNIT_1
name: SkyInv
ports:
  VDD: ps,s
  VSS: ns,s
  i: ps,g
  o: ps,d
"""
with open("data/sky130nm/example_inverter_manual.schem.yaml", 'w') as file:
        file.write(example_inverter_manual_yaml)
# -

# ### Automatically mapping layout instances to the YAML

example_inverter_schematic_editor = gph.hdl21_module_to_schematic_editor(
    module=SkyInv,
    yaml_schematic_file_name="data/sky130nm/example_inverter_auto.schem.yaml",
)
example_inverter_schematic_editor.visualize()

example_inverter_layout = "data/sky130nm/example_inverter_auto.layout.yaml"
example_inverter_schematic_editor.instantiate_layout(example_inverter_layout, default_router="get_bundle", default_cross_section="xs_metal1")
c = gf.read.from_yaml(example_inverter_layout)
c.plot()

# ## (WIP, future integrations) Setting up the required tools: `xschem`, `volare` and the `sky130nm` PDKs
#
# I will say early on, you would benefit from working in a UNIX or Debian Linux environment. Most EDA tools either proprietary or open-source only work within these operating systems or a Docker environment of these operating systems.
#
# There are multiple ways to get started with the environment, you could, for example:
# 1. Work within a [IIC-OSIC-TOOLS docker environment](https://github.com/iic-jku/IIC-OSIC-TOOLS). There is also a nice [OSIC-multitool project](https://github.com/iic-jku/osic-multitool) specifically for SKY130nm designs.
# 2. Install these tools natively in your operating system. You could follow one of these tutorials, for example:
#     * https://www.engineerwikis.com/wikis/installation-of-xschem
#     * https://www.youtube.com/watch?v=jXmmxO8WG8s&t=7s
#
# This tutorial assumes the tools have already been installed in a local operating system. We will use some design files generated by these toolsets as guided by this [OSIC-Multitool example](https://github.com/iic-jku/osic-multitool/tree/main/example/ana). Note we recommend cloning the `gplugins` repository.

import pathlib
from gplugins.spice import parse_netlist as spice


# Our example files are under the directory of `gplugins/notebooks/data`, let's extract our SPICE declaration:

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None


# Let's list the files we're going to be reading into:

inverter_spice_file = pathlib.Path("data") / "sky130nm" / "inv.sch"

# Let's extract our raw spice netlist. An important aspect to understand is that unfortunately, every SPICE tool developed their own file format. So in this sense, netlist parsing function is implemented according to the type of spice toolset that has generated this netlist.
#
# We aim to support:
#
# - `xschem`
# - `lumerical?`

inverter_spice_text = read_file(inverter_spice_file)
inverter_spice_text

spice.netlist_text

elements, _, _ = spice.parse_netlist_and_extract_elements(netlist_text=spice.netlist_text, spice_type="lumerical")
elements

# So we can use our `netlist` parsing function to convert this to a compatible netlist for gdsfactory plugins into the extracted elements and the extracted connections:

inverter_netlist_elements, inverter_netlist_connections, _ = spice.parse_netlist_and_extract_elements(netlist_text=inverter_spice_text, spice_type="xschem")
inverter_netlist_elements, inverter_netlist_connections, _

# ### Automated schematic-driven-layout

# We have now extracted our spice elements and our connectivity. Let's explore what we have there:

# + active=""
# # Current TODOs
#
# 1. Update the extraction function to add xschem compatibility.
# 2. Create the mapping between the extracted netlist and the corresponding SKY130nm elements.
# -




