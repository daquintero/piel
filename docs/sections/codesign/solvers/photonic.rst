Optical
-------------

This references are based on Chrostowski's *Silicon Photonic Design* Chapter 2:

If we observe into the cross section of a waveguide towards the transverse propagation of light, we can observe that the optical wave modes are propagating invariantly in time. Maxwell's equvations time-harmonic solutions are solved into the frequency domain by these solvers. this solution is for a specific frequency. To analyse a broadband pulse at multiple wavelength modes, the mode solver has to run a frequency sweep.

Full solutions include Finite Element Method and Finite Difference algorithms. Approximated solutions include the Effective Index Method.

-  Finite Element Method solvers are good for unstructured multi-geometrical shapes or 3D modelling.
-  Finite Difference solvers are good for high-index contrast structures.


3D FDTD
~~~~~~~~~~~~~~~~~~~~~

Three-dimensional Maxwell's equations are solved by this tool. We explore light interaction with sub-wavelength features such as those that can be lithographically fabricated in a silicon chip. Accuracy converges to an exact solution as the mesh size is reduced, hence this is the most accurate solver. Simulation time step can be sub-femtosecond so large processing units are required for large scale use. It accurately models a wide-optical-bandwidth response, and can generate scattering parameters for its simulations.

Used to calculate:

- Waveguide bends
- Coupling coefficients between waveguides
- Bragg reflections and transmission
- Field profiles and reflections

Procedure:

1. Generate material definitions
2. Generate optical structure
3. Simulation volume is defined. Mesh :math:`\Delta x` >14-18 points per wavelength. Simulation time :math:`\propto \frac{1}{\Delta x^4}`.
4. Define matching boundary layers.
5. Optical sources are injected. Because FDTD is a time-to-frequency simulation, a short pulse length translates to a broad optical spectrum.
6. Monitors are used to measure electronic and optical field quantities. Fundamental monitor is time-domain monitor, and this is wavelength dependent.
7. Perform convergence simulations to make sure there are no numerical errors.
8. Perform analysis on the signal profiles.

Common solvers that implement this:
- ``Tidy3D``
- ``Lumerical FDTD``.


2.5D with Effective Index FDTD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can be similar to 3D FDTD when there is no coupling between TE and TM modes.

Used to calculate:

- Planar photonic integrated circuits
- Waveguide systems
- Ring resonators
- Planar omi-directional propagation without assumptions like an optical -axis

Common solvers that implement this:
- ``Tidy3D``
- ``Lumerical MODE``.


Eigenmode Expansion Method EME
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The propagation of light in local fields is decomposed into modes at that position. Uses the scattering-parameters to connect the next section of the device which makes it inherently bi-directional as it includes both forward and backwards propagation for an infinite number of modes.

Used to calculate:

- Waveguide sections
- Directional couplers
- Interferometer region in a MMI


Electronic \& Optical
-----------------------

This is the type of modelling that we particularly care about. However, it is important to note the importance and effect of each individual simulation implementation accordingly. This is where the multi-physical component-level domain simulations are integrated.
