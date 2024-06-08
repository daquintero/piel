A primer on common photonic-electronic systems
-----------------------------------------------

We will explore some common photonic-electronic systems and discuss the design flows involved afterwards.

An Opto-electronic System
^^^^^^^^^^^^^^^^^^^^^^^^^^

A relatively-common system is one where an electronic signal is generated from a photonic device. This involves the readout of photodetectors such as silicon photodiodes, photoresistors, or more-quantum-focused single-photon detectors such as superconducting nanowires. TODO ref.

.. figure:: ../../../_static/img/sections/codesign/detector_photonic_system.png
   :alt: Example detector integrated photonics-electronics system

For many of these photodetectors, they need to be biased or amplified with analogue or RF electronics. This is often discretized into digital electronics through an equivalent ADC.

These systems are commonly used as sensing-based detectors, and can be used for very precise metrology. TODO add ref.

An Electro-Optic System
^^^^^^^^^^^^^^^^^^^^^^^^^

Another common system in the integrated photonics domain is an electronically-controlled photonic system. In this case, en electronic signal can control an optical signal such as a laser pulse, continuous laser, or more-quantum-focused single-photon and change it's properties such as phase or amplitude.

These types of systems are more common in datacenters, photonics-driven-HPC, neuro-morphic photonic computation, and quantum photonic systems. TODO ref.

.. figure:: ../../../_static/img/sections/codesign/switch_photonic_system.png
   :alt: Example switched integrated photonics-electronics system

To optimise power consumption, it might be desired to design custom digital logic, with custom amplifiers and specifications, for a application-specific integrated photonic device.

In an integrated-photonics platform, it is common to use a Mach-Zehnder interferometer as en electro-optic switch. This is by the use of an active electro-optic or thermo-optic phase shifter on one interferometer arm, can be use to route a photonic path output or not.

A Concurrent Photonic-Electronic System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the most interesting ones in my opinion. Concurrent photonic-electronic systems involve the interaction between optical signals and electronics signals in real-time, with feedback and feedforward. This is currently at the cusp of scientific development of current technology.

However, these systems are very hard to simulate fully. It involves the time-domain interaction of analogue, digital and photonic signals in order to model a specific behaviour. These are the type of systems we are interested in modelling through `piel`. The flexibility of open-source software enables us to create these type of modelling systems in a way that closed-source software can make it very difficult to extend.

.. image:: ../../../_static/img/sections/codesign/feedback_photonic_system.png
   :alt: Example feedback integrated photonics-electronics system
