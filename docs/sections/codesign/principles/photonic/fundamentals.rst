Let's review some theory on photonic and RF network design based on
Chrostowski's *Silicon Photonic Design* and Pozar's *Microwave Engineering*. Optical and radio-frequency signals are both electromagnetic, within different orders of magnitude of frequency. The electro-magnetic theory principles are very similar, albeit there are different terminologies for some very similar things. We need to understand crucial differences between photonic and RF network design in order to properly design coupled electrical-photonic systems.

It is also important to understand electromagnetic propagation representation when we consider the simulation methodologies implemented by different solvers.


What is an electromagnetic wave?
---------------------------------------

A Scottish man named James Clark Maxwell knew, and because of him so do we:

.. math::

    \begin{align}
        \nabla \times \mathcal{E} = \frac{ - \delta \mathcal{B} }{\delta t} - \mathcal{M} \\
        \nabla \times \mathcal{H} = \frac{\delta \mathcal{D}}{\delta t} - \mathcal{J} \\
        \nabla \cdot \mathcal{H} = \rho \\
        \nabla \cdot \mathcal{B} = 0
    \end{align}


Note the units:

-  :math:`\mathcal{E}` electric field in volts per meter :math:`V/m`
-  :math:`\mathcal{H}` magnetic field in amperes per meter :math:`A/m`
-  :math:`\mathcal{D}` electric flux density in Coulombs per meter squared :math:`C/m^2`
-  :math:`\mathcal{B}` magnetic flux density in Webers per meter :math:`Wb/m`
-  :math:`\mathcal{J}` electric current density in amperes per meter squared :math:`A/m^2`
-  :math:`\mathcal{\rho}` electric charge density in Couloms per meter cubed :math:`C/m^3`


A sinusodial electric field polarised in the :math:`x` direction can be generically written as:

.. math::

    \mathcal{E}(x,y,z,t) = \hat{x} A(x,y,z) cos(\omega t + \phi)

where :math:`A(x,y,z)` is the amplitude function dependent on spatial dimensions, :math:`\omega` radian frequency, and :math:`\phi` phase reference shift from the wave at time :math:`t=0`. The wave is polarised because only the :math:`\hat{x}` component of the amplitude function is relevant.


Can we simplify this for our applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an isotropic, homogeneous medium where the signal wavelengths do not interact with nonlinear material properties, we can solve Maxwell's curl equations in a phasor form known as the Helmholtz equations. This is not always valid for photonic network analysis as there can nonlinear material interactions such as spontaneous four-wave mixing and we will explore this afterwards, and is an approximation more common to radio-frequency analysis.

.. math::

    \begin{align}
        \nabla \times E = - j \omega \mu E \\
        \nabla \times H = - j \omega\epsilon E
    \end{align}

These equations can be thought of as simultaneous equations. With some vector calculus, they can be solved into:

.. math::

    \begin{align}
        \nabla^2 E + \omega^2 \mu \epsilon E = 0  \\
        \nabla^2 H + \omega^2 \mu \epsilon H = 0
    \end{align}


The wavenumber constant :math:`k = \omega \sqrt{\mu\epsilon}` relates the material dielectric constant :math:`\epsilon` and magnetic permeability :math:`\mu` to a travelling plane electromagnetic wave in any medium. This is also called the propagation constnat to describe how the wave changes with distance :math:`z`.

A general solution
'''''''''''''''''''

.. math::

    \begin{equation}
    \nabla^2 E + k^2 E = \frac{d^2 E}{d^2 x^2} + \frac{d^2 E}{d^2 y^2} + \frac{d^2 E}{d^2 z^2} + k^2E = 0
    \end{equation}

We can say that this definition is valid for every spatial component of the field.

Solving the above equation as a partial-differential-equation with separation of variables as done by *Microwave Engineering* by Pozar we can derive that the total wave propagation constant is directionally composed:

.. math::

    \begin{equation}
    k_x^2 + k_y^2 + k_z^2 = k^2
    \end{equation}

We can write the electric field component in the :math:`x`-direction as a function of the electric field in space coordinates (:math:`x`, :math:`y`, :math:`z`) due to the :math:`k(x,y,z)`

Helmholtz's on a lossless waveguide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume we have a lossless waveguide transmitting a photonic or radio-frequency electromagnetic wave. It has a uniform cross-section in the :math:`x` and :math:`y` plane, and the wave propagates transversely in the :math:`z` dimension. Because of the uniform cross-section, the electromagnetic fields don't change in the :math:`x` and :math:`y` directions inside the waveguide. This means that :math:`\frac{d}{dx} = \frac{d}{dy} = 0` in this solution.

If we assume one-dimensional signal propagation in the :math:`z` direction, and just considering a signal amplitude in the :math:`x` dimension.

.. math::

    \begin{equation}
        \frac{d^2 E_x}{dz^2} + (\omega^2\mu\epsilon) E_x = 0
    \end{equation}

With this simplification, we can derive the harmonic solution at frequency :math:`\omega` :

.. math::

    E_x(z) = E^+ e^{-jkz} + E^- e^{jkz}

In time, the solution is:

.. math::

    \begin{equation}
        \mathcal{E}_x(z,t) =  E^+ \cos(\omega t-kz) + E^- \cos(\omega t+kz)
    \end{equation}

The :math:`E^+` refers to the forward-propagating wave amplitude, and :math:`E^-` as the back-propagating amplitude of the wave. If we consider a fixed-point on the wave :math:`\omega t-kz = \text{constant}`, then for increasing time, the :math:`z` position must also increase which is why this wave is forward-propagating. This is reciprocal for the backward propagating wave :math:`E^-` with a :math:`\omega t+kz = \text{constant}` wave definition. This definition of wave velocity in terms of a fixed-point in the wavefront it called *phase velocity* and is formally defined as:

.. math::

    \begin{equation}
        v_p = \frac{dz}{dt} = \frac{wt - \text{constant}}{k} = \frac{w}{k} = \frac{1}{\sqrt{\mu \epsilon}}
    \end{equation}

The physical distance between two peaks of a sinusodial wave is called the wavelength of the wave and is defined by:

.. math::

    \begin{equation}
        \lambda = \frac{2\pi}{k}
    \end{equation}

It is this wavelength that determines the color of light. However, in a normal pulse of bright light, there is a spectrum of wavelengths contained within it. The physical interaction at the dimensions of the wavelength also lead to a number of quantum light-matter interactions which are important when considering nonlinear material effects.

This also means that the propagation constant can be defined in relation to wavelength:

.. math::

    \begin{equation}
        k = \frac{2\pi}{\lambda}
    \end{equation}

This is sometimes interesting in analysing dispersive photonic systems.

We will explore how to analyse this in a integrated silicon waveguide afterwards.

The Definition of Phase
-------------------------

If we are considering two identical frequency sinusoidal waves at a particular instance in time, their phase differential corresponds to the difference between their wavefronts position.

Consider we have a plane electro-magnetic wave propagating in time :math:`t`. What makes it a plane wave is that the electric and magnetic fields are transverse to the direction of propagation :math:`z`, both electric and magnetic fields only exist in a direction (say :math:`x` or :math:`y`), and their field magnitude is constant in the :math:`z` direction.

TODO add picture.

.. math::

    \begin{align}
        E = E_0 e^{j(kz \pm \omega t)} \\
        H = H_0 e^{j(kz \pm \omega t)} \\
    \end{align}


Remember that a sinusodial signal is defined by Euler's formula, so we can work in terms of phasor notation.

.. math::

    \begin{equation}
        e^{jx} = cos(x) + j sin(x)
    \end{equation}

Reed and Knights describe the definition of polarisation succinctly:

    It is the direction of the electric field associated with the propagating wave.


Making Waves Interfere
--------------------------

Let's assume we have two waves aligned in space in terms of polarisation. They are also *coherent* waves, which means that they have a constant phase :math:`kz \pm \omega t` relationship. This tends to mean that the waves come from an equivalent source. If these two waves are coincident in a point in space, the electric and magnetic fields of the waves add together.


Guided Waves
-------------------

TODO add image

At secondary school, we learn that if we have an interface of two optical materials with different refractive indices :math:`n_1` and :math:`n_2`, and light rays with angles of incidence :math:`\theta_1` and refraction :math:`\theta_2`, then we can relate the rays angles according to Snell's law:

.. math::

    \begin{equation}
        n_1 sin(\theta_1) = n_2 sin(\theta_2)
    \end{equation}

Light can propagate at the interface of the two materials at a critical angle :math:`\theta_c` where the first material's refractive index is higher than the second interface material. This equation has a valid solution only when :math:`n_1 > n_2`

.. math::

    \begin{equation}
        sin(\theta_c) = \frac{n_2}{n_1}
    \end{equation}

Any incident light angles greater than the critical angle at this first boundary material get totally internally reflected back into the material.

However, we're grown ups now, we can think about this in terms of waves too.

A transverse electromagnetic wave (TEM) describes a wave where electric and magnetic components of the wave are propagating orthogonally to each other. We can describe waves according to the direction of their electromagnetic components. A transverse electric (TE) wave has the electric field polarisation directed orthogonal to the incidence direction of the wave. A transverse magnetic (TM) wave has the magnetic field polarisation directed orthogonal to the incidence direction of the wave.

We often care about the power of the reflected and transmission of the waves at these interfaces. We describe this in terms of a Pointing vector, commonly denoted as :math:`S` with :math:`\frac{W}{m^2}` units to describe intensity per area. This wave is propagating through a medium with a given impedance :math:`Z` which in this electromagnetic regime is related to the dielectric and permeability material properties.

.. math::

    \begin{equation}
        S = \frac{1}{Z} E^2 = \sqrt{\frac{\epsilon}{\mu}} E^2
    \end{equation}

The reflectance of an incident wave with power :math:`S_i` and reflected to a wave with power :math:`S_r` can be described in terms of the waves:

.. math::

    \begin{equation}
    R = \frac{S_r}{S_i} = \frac{E_r^2}{E_i^2}
    \end{equation}


Towards Waveguides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a waveguide where an electromagnetic wave propagates through a total internal reflection in a medium with refractive index :math:`n`, we can describe the following relationship for the propagation constant:

.. math::

    \begin{equation}
    k = n k_0
    \end{equation}

Where the free space propagation constant is defined a in relation to the free-space wavelength :math:`lambda_0`:

.. math::

    \begin{equation}
    k_0 = \frac{2\pi}{\lambda_0}
    \end{equation}

TODO image here

If we have a waveguide with a core defined by a :math:`n_1` refractive index and a height :math:`h` in the :math:`y` direction, for a wave propagating in the :math:`z` direction, we can decompose the ideal trigonometric propagation of the wave into directional propagation constants:

.. math::

    \begin{align}
    k_z = n_1 k_0 sin(\theta_1) \\
    k_y = n_1 k_0 cos(\theta_1)
    \end{align}

If we look into the waveguide, we would be observing the :math:`y` component of the wave as it reflects and a standing wave between its components.

Let's consider a full-round trip of our wave as it reflects in the core. The transvered distance of the wave is :math:`2h`. We know, fundamentally, that the propagation constant is related to differential of the phase of the wave propagating in :math:`z`:

.. math::

    \begin{equation}
        \frac{\delta \phi}{\delta z} = k
    \end{equation}

Which means that for a 3D wave, if we integrate over a length component in the :math:`y` direction component only, we know that:

.. math::

    \begin{equation}
        \phi_h = 2 k_y h = 2 k_0 n_1 h cos(\theta_1)
    \end{equation}

We also know that there are some phase changes introduced at each interface denoted :math:`\phi_{int}` due to Fresnel's equations but maybe in the future I'll get to that. We also know that the total phase shift introduced by the propagation in the waveguide must be a multiple of :math:`2\pi` (so that it keeps being a wave). This allows us to create the following relationship:

.. math::

    \begin{equation}
        2 k_0 n_1 h cos(\theta_1) - \phi_{int} = 2m \pi
    \end{equation}

Because :math:`m` is an integer, there are only a discrete set of angles at which this is valid. This is what we refer to when we talk about the mode of propagation of the wave for a mode number :math:`m`.

Reed and Knights *Silicon Photonics* derive this further, but we can solve for the maximum mode number :math:`m` possible in a waveguide:

.. math::

    \begin{equation}
        m_{max} = \frac{k_0 n_1 h cos(\theta_c)}{\pi}
    \end{equation}


It is really important to consider how the most change when we design a photonic circuit, as whatever mismatch we might have between our components means that our circuit would radiate the signal away. As such, it is very important to account for mode perturbations from electronic control of our devices.



Understanding our Materials
-----------------------------


When doing photonic design, a common and very popular material is
silicon. However, we need to understand how our pulses propagate along
it. Silicon refractive index :math:`n_{Si}` is wavelength dependent and
can be described by Sellmeier equation:

.. math::

    \begin{equation}
        n^2 (\lambda) =  \eta + \frac{A}{\lambda^2} + + \frac{B \lambda_1^2}{\lambda^2 - \lambda_1^2}
    \end{equation}


In a dielectric material like silicon, the applied electric field can align electric charges in atoms and amplifies the total electric flux density in units :math:`C/m^2`. The polarization by an applied electric field can be considered a capacitance variation effect. A real example of this is ceramic derate their capacitance value based on the applied DC electric field.

.. math::

    \begin{equation}
    \mathcal{D} = \epsilon_0 E + P_e
    \end{equation}


The polarization :math:`P_e` is related to the electric field by the electric susceptibility :math:`\chi_e` which is just a complex form of the dielectric constant :math:`\epsilon`:

.. math::

    \begin{align}
    P_e = \epsilon_0 \chi_e \\
    D = \epsilon_0 (1 + \chi_e) E = \epsilon E \\
    \end{align}

A general relationship (normally simplified for silicon, and instead more valid for other materials) relates the electric flux density to the electric field applied through a spatially variating electric field and dielectric constant:

.. math::

    \begin{equation}
    \begin{bmatrix}
        D_x \\
        D_y \\
        D_z \\
    \end{bmatrix} =
    \begin{bmatrix}
        \epsilon_{xx} & \epsilon_{xy} & \epsilon_{xz} \\
        \epsilon_{yx} & \epsilon_{yy} & \epsilon_{yz} \\
        \epsilon_{zx} & \epsilon_{zy} & \epsilon_{zz} \\
    \end{bmatrix}
    \begin{bmatrix}
        E_x \\
        E_y \\
        E_z \\
    \end{bmatrix} =
    [\epsilon]
    \begin{bmatrix}
        E_x \\
        E_y \\
        E_z \\
    \end{bmatrix}
    \end{equation}

In this sense, we can think of electric fields propagating in a dielectric material such as our silicon waveguides. It is important to note that our electric fields are vectorial, and tensor materials operate on them.

Propagation & Dispersion
-----------------------------

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
----------------------

In a photonic waveguide:

-  Photon absorption due to metal in near the optical field.
-  Sidewall scattering loss, and rough sidewalls introduce reflections
   and wavelength dependent phase perturbations
-  Loss due to doped or an absorptive material in the waveguide

You can reduce loss by having multi-mode wider waveguides. When we apply
different electronic states to our phase shifter, we are changing the
optical material parameters. As such, we are also affecting the
time-delay of our pulse propagation.
