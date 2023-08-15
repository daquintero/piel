Theory Review
-------------

Let's review some theory on photonic and RF network design based on
Chrostowski's *Silicon Photonic Design* and Pozar's *Microwave Engineering*. Optical and radio-frequency signals are both electromagnetic, within different orders of magnitude of frequency. The electro-magnetic theory principles are very similar, albeit there are different terminologies for some very similar things. We need to understand crucial differences between photonic and RF network design in order to properly design coupled electrical-photonic systems.

It is also important to understand electromagnetic propagation representation when we consider the simulation methodologies implemented by different solvers.


What is an electromagnetic wave?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Scottish man named James Clark Maxwell knew, and because of him so do we:

.. math::

    \begin{align}
        \nabla \times \mathcal{E} = \frac{ - \delta \mathcal{B} }{\delta t} - \mathcal{M} \\
        \nabla \times \mathcal{H} = \frac{\delta \mathcal{D}}{\delta t} - \mathcal{J} \\
        \nabla \cdot \mathcal{H} = \rho \\
        \nabla \cdot \mathcal{B} = 0
    \end{align}


Can we simplify this for our applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an isotropic, homogeneous medium where the signal wavelengths do not interact with nonlinear material properties, we can solve Maxwell's curl equations in a phasor form known as the Helmholtz equations. This is not always valid for photonic network analysis as there can nonlinear material interactions such as spontaneous four-wave mixing and we will explore this afterwards.

.. math::

    \begin{align}
        \nabla \times E = - jw \mu E \\
        \nabla \times H = - jw \epsilon E
    \end{align}

These equations can be thought of as simultaneous equations. With some vector calculus, they can be solved into:

.. math::

    \begin{align}
        \nabla^2 E + w^2 \mu \epsilon E = 0  \\
        \nabla^2 H + w^2 \mu \epsilon H = 0
    \end{align}


The wavenumber constant :math:`k = w\sqrt{\mu\epsilon}` relates the material dielectric constant :math:`\epsilon` and magnetic permeability :math:`\mu` to a travelling plane electromagnetic wave in any medium.


Helmholtz's on a lossless waveguide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume we have a lossless waveguide transmitting a photonic or radio-frequency electromagnetic wave. It has a uniform cross-section in the :math:`x` and :math:`y` plane, and the wave propagates transversely in the :math:`z` dimension. Because of the uniform cross-section, the electromagnetic fields don't change in the :math:`x` and :math:`y` directions inside the waveguide. This means that :math:`\frac{d}{dx} = \frac{d}{dy} = 0` in this solution.

If we assume one-dimensional signal propagation in the :math:`z` direction, and just considering a signal amplitude in the :math:`x` dimension.

.. math::

    \begin{equation}
        \frac{d^2 E_x}{dz^2} + (\omega^2\mu\epsilon) E_x = 0
    \end{equation}

With this simplification, we can derive the harmonic solution at frequency :math:`\omega` :

.. math::

    E_x(0 < z < l) = E^+ e^{-jkz} + E^- e^{jkz}


In time, the solution is:

.. math::

    \begin{equation}
        \mathcal{E}_x(z,t) =  E^+ \cos(\omega t-kz) + E^- \cos(\omega t+kz)
    \end{equation}

The :math:`E^+` refers to the forward-propagating wave amplitude, and :math:`E^-` as the back-propagating amplitude of the wave. If we consider a fixed-point on the wave :math:`\omega t-kz = \text{constant}`, then for increasing time, the :math:`z` position must also increase which is why this wave is forward-propagating. This is reciprocal for the backward propagating wave :math:`E^-` with a :math:`\omega t+kz = \text{constant}` wave definition. This definition of wave velocity in terms of a fixed-point in the wavefront it called *phase velocity* and is formally defined as:

.. math::

    \begin{equation}
        v_p = \frac{dz}{dt} = \frac{wt - \text{constant}}{k} = \frac{w}{k} = \frac{1}{\mu \epsilon}
    \end{equation}

We will explore how to analyse this in a integrated silicon waveguide afterwards.




The Definition of Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we are considering two identical frequency sinusoidal waves at a particular instance in time, their phase differential corresponds to the difference between their wavefronts position.

Consider we have a one dimensional optical wave propagating in time :math:`t`:

.. math::

    \begin{equation}
        E = E_0 e^{j(kz \pm \omega t)}
    \end{equation}

Remember that a sinusodial signal is defined by Euler's formula:

.. math::

    \begin{equation}
        e^{ix} = cos(x) + j sin(x)
    \end{equation}


Understanding our Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


When doing photonic design, a common and very popular material is
silicon. However, we need to understand how our pulses propagate along
it. Silicon refractive index :math:`n_{Si}` is wavelength dependent and
can be described by Sellmeier equation:

.. math::

    \begin{equation}
        n^2 (\lambda) =  \eta + \frac{A}{\lambda^2} + + \frac{B \lambda_1^2}{\lambda^2 - \lambda_1^2}
    \end{equation}


Propagation & Dispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


One important aspect we care about when doing co-simulation of
electronic-photonic networks is the time synchronisation between the
physical domains.

In a photonic waveguide, the time it takes for a pulse of light
with wavelength :math:`\lambda` to propagate through it is dependent on
the group refractive index of the material at that waveguide
:math:`n_{g}`. This is because we treat a pulse of light as a packet of
wavelengths.

.. math::

    \begin{equation}
    v_g (\lambda) = \frac{c}{n_{g}}
    \end{equation}`

If we wanted to determine how long it takes a single phase front of the
wave to propagate, this is defined by the phase velocity :math:`v_p`
which is also wavelength and material dependent. We use the effective
refractive index of the waveguide :math:`n_{eff}` to calculate this,
which in vacuum is the same as the group refractive index :math:`n_g`,
but not in silicon for example. You can think about it as how the
material geometry and properties influence the propagation and phase of
light compared to a vacuum.

.. math::

    \begin{equation}
    v_p (\lambda) = \frac{c}{n_{eff}}
    \end{equation}

Formally, in a silicon waveguide, the relationship between the group index and the effective
index is:

.. math::

    \begin{equation}
    n_g (\lambda) = n_{eff} (\lambda) - \lambda \frac{ d n_{eff}}{d \lambda}
    \end{equation}

If we want to understand how our optical pulses spread throughout a
waveguide, in terms of determining the total length of our pulse, we can
extract the dispersion parameter :math:`D (\lambda)`:

.. math::

    \begin{equation}
    D(\lambda) = \frac{d \frac{n_g}{c} }{d \lambda} = - \frac{\lambda}{c} \frac{d^2 n_{eff}}{d \lambda^2}
    \end{equation}


Sources of Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a photonic waveguide:

-  Photon absorption due to metal in near the optical field.
-  Sidewall scattering loss, and rough sidewalls introduce reflections
   and wavelength dependent phase perturbations
-  Loss due to doped or an absorptive material in the waveguide

You can reduce loss by having multi-mode wider waveguides. When we apply
different electronic states to our phase shifter, we are changing the
optical material parameters. As such, we are also affecting the
time-delay of our pulse propagation.
