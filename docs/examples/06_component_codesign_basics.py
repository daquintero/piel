# # Component Codesign Basics

# When we have photonic components driven by electronic devices, there is a scope that we might want to optimise certain devices to be faster, smaller, or less power-consumptive. It can be complicated to do this just analytically, so we would like to have the capability of integrating our design software for each of our devices with our simulation software of our electronics. There might be multiple  software tools to design different devices, and the benefit of integrating these tools via open-source is that co-design becomes much more feasible and meaningful.
#
# In this example, we will continue exploring the co-design of a thermo-optic phase shifter in continuation of all the previous examples. However, this time, we will perform some optimisation in its design parameters and related. We will use the `femwell` and `devsim` packages that are part of the `GDSFactory` suite.
