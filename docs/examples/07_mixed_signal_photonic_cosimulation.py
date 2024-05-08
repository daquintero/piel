# # Mixed-Signal & Photonic Cosimulation

# ## Logical Electronic-Photonic Co-Design
#
# In this example, we will demonstrate how we can take a logical optical function and implement the electronic logic required to control it. 

# ### Extracting Cross-Bar Optical Switch States

# One of the main complexities of electronic-photonic codesign is determining the required control interface between the designs. Let's consider we have a specific instantiation of a 2x2 Mach-Zehnder interferometer switch, let's extract the corresponding cross and bar phase state implementation. We need to know what phase to apply within a larger switch lattice accordingly in order to model the switch logic operation.



# #### Automation

# Now, these transient simulations are something you might want to very configure depending on the type of signals that you might want to verify. However, we can provide some basic parameterised simple functions such as step responses and so on. So instead of having to write everything above, you can also just run the following, for example: WIP

# One desired output of an electrical model simulation is an extraction of the power consumption of the circuit. Fundamentally, this is dependent on the time and the operation performed. Hence, to estimate an average power consumption, it is more effective to define the power consumption of a particular operation, and extract the power consumption for the frequency at which this operation is performed.
#
# In this case, we are defining the energy of the operation at particular nodes of the circuit. For example, we know a resisitve heater will dissipate all of its current consumption as thermal power. However, we also need to evaluate the whole circuit. We can know how much energy our DC or RF power supply is providing by measuring the voltage and current supplied accordingly. In a digital circuit, depending on the frequency of the operation, we know how often there is a signal rise and fall, in both cases forcing digital transistors to operate in the linear regime and consuming more power than in saturation mode. We also need to account the range of time the signals are in saturation mode, as even in CMOS idle state there is a minimal power consumption that is important as the circuit scales into VLSI/EP.
#
# Note that through the SPICE simulations, we can extract the energy required at each operation with greater accuracy than analytically and the complexity of this is configuring the testbench appropriately in order to account for this.

#

#

#
