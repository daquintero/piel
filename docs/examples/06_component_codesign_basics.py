# # Component Codesign Basics

# When we have photonic components driven by electronic devices, there is a scope that we might want to optimize certain devices to be faster, smaller, or less power-consumptive. It can be complicated to do this just analytically, so we would like to have the capability of integrating our design software for each of our devices with the simulation software of our electronics. There might be multiple software tools to design different devices, and the benefit of integrating these tools via open-source is that co-design becomes much more feasible and meaningful.
#
# In this example, we will continue exploring the co-design of a thermo-optic phase shifter in continuation of all the previous examples.
#
# This example consists of a few things:
# * Connect a `gdsfactory` component to a `femwell` model, and relate the corresponding metrics.
# * Extract a `hdl21 SPICE`model from the generated component we can use in circuit simulators.
# * Simulate the mode profiles via `Tidy3D FDTD` and extract the `S-Parameters` from the layout.
# * Create a flow where variations on both electronic and photonic parameters can be quantified through this model.
#
# Good examples of sections of this flow are:
# * [gplugins-femwell HEAT](https://gdsfactory.github.io/gplugins/notebooks/femwell_02_heater.html)
# * [gplugins-tidy3d MODE](https://gdsfactory.github.io/gplugins/notebooks/tidy3d_01_tidy3d_modes.html)
# * [gplugins-tidy3d FDTD](https://gdsfactory.github.io/gplugins/notebooks/tidy3d_00_tidy3d.html)
# * Some of the previous examples in `piel`


# ## Starting from our heater geometry
#
# We will begin by extracting the electrical parameters of the basic `gf.components.straight_heater_metal_simple` which we have been using in our Mach-Zehnder MZI2x2 from the example provided in [`femwell`](https://helgegehring.github.io/femwell/photonics/examples/metal_heater_phase_shifter.html).
#
# We will create a function where we change the width of the heater, and we explore the change in resistance, but also in thermo-optic phase modulation efficiency. So, we want to have some easy functions that allow us to easily mesh a component because we plan to do a few variations.
