# # Analogue Circuit Layout & Simulation

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

# ## Schematic-Driven-Layout

# A common analogue design flow is called schematic-driven-layout. What this entails, fundamentally, is that we design a circuit through a schematic, and then use that schematic to instruct, extract, constrain, and/or verify our circuit chip layout. This flow uses layout elements that are connected or tied to schematic symbols, and unique names that allow for identification and connectivity relationship.
#
# TODO ADD SOME REFERENCES EXPLAINING SCHEMATIC DRIVEN LAYOUT.
#
# In an open-source flow, this could be, for example, demonstrated by creating a circuit using the `hdl21 schematic` tools. Each symbol would reference a specific `PCell` in the PDK. Now, we would use this individual element cell name to connect and extract to the `SPICE` model and also to the `layout` GDS cell. This allows us to connect to the separate tools for simulation and layout.
#
# Say, we can then extract a netlist from the schematic with individual cell names and PDK cells identifiers. We could in `gdsfactory` map these PDK cell identifiers to instantiate the elements in a layout. We can then use this instantated cells to perform some automatic or

# ## Layout-Extracted-Schematic

# Say we have a layout of a chip with several cells.
