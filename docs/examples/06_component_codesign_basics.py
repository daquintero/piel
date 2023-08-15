# # Component Codesign Basics

# When we have photonic components driven by electronic devices, there is a scope that we might want to optimise certain devices to be faster, smaller, or less power-consumptive. It can be complicated to do this just analytically, so we would like to have the capability of integrating our design software for each of our devices with our simulation software of our electronics. There might be multiple  software tools to design different devices, and the benefit of integrating these tools via open-source is that co-design becomes much more feasible and meaningful.
#
# In this example, we will continue exploring the co-design of a thermo-optic phase shifter in continuation of all the previous examples. However, this time, we will perform some optimisation in its design parameters and related. We will use the `femwell` package that is part of the `GDSFactory` `GPlugins` suite.

# ## Theory Review
#
# Let's review some theory on photonic network design based on Chrostowski's *Silicon Photonic Design*:

# When doing photonic design, a common and very popular material is silicon. However, we need to understand how our pulses propagate along it. Silicon refractive index $n_{Si}$ is wavelength dependent and can be described by Sellmeier equation:
#
# \begin{equation}
# n^2 (\lambda) =  \eta + \frac{A}{\lambda^2} + + \frac{B \lambda_1^2}{\lambda^2 - \lambda_1^2}
# \end{equation}

# ### Propagation \& Dispersion
#
# One important aspect we care about when doing co-simulation of electronic-photonic networks is the time synchronisation between the physical domains. In a waveguide, the time it takes for a pulse of light with wavelength $\lambda$ to propagate through it is dependent on the group refractive index of the material at that waveguide $n_{g}$. This is because we treat a pulse of light as a packet of wavelengths.
#
# \begin{equation}
# v_g (\lambda) = \frac{c}{n_{g}}
# \end{equation}
#
# If we wanted to determine how long it takes a single phase front of the wave to propagate, this is defined by the phase velocity $v_p$ which is also wavelength and material dependent. We use the effective refractive index of the waveguide $n_{eff}$ to calculate this, which in vacuum is the same as the group refractive index $n_g$, but not in silicon for example. You can think about it as how the material geometry and properties influence the propagation and phase of light compared to a vacuum.
#
# \begin{equation}
# v_p (\lambda) = \frac{c}{n_{eff}}
# \end{equation}
#
# Formally, the relationship between the group index and the effective index is:
#
# \begin{equation}
# n_g (\lambda) = n_{eff} (\lambda) - \lambda \frac{ d n_{eff}}{d \lambda}
# \end{equation}
#
# If we want to understand how our optical pulses spread throughout a waveguide, in terms of determining the total length of our pulse, we can extract the dispersion parameter $D (\lambda)$:
#
# \begin{equation}
# D(\lambda) = \frac{d \frac{n_g}{c} }{d \lambda} = - \frac{\lambda}{c} \frac{d^2 n_{eff}}{d \lambda^2}
# \end{equation}

#

# When we apply different electronic states to our phase shifter, we are changing the optical material parameters. As such, we are also affecting the time-delay of our pulse propagation.

# ## Start from Femwell `TiN TOPS heater` example
#
# We will begin by extracting the electrical parameters of the basic `TiN TOPS heater` example provided in [`femwell`](https://helgegehring.github.io/femwell/photonics/examples/metal_heater_phase_shifter.html). We will create a function where we change the width of the heater, and we explore the change in resistance, but also in thermo-optic phase modulation efficiency.
