Mixed-Signal Electronic-Photonic Co-Simulation
==============================================

Time Synchronisation Complexities
---------------------------------

One of the main complexities of this type of simulation software is the mapping
of different dimensions of time. For example, the time of the laser wave
is has its electromagnetic components changing in the femtosecond
regime, whereas digital signals may be changing in the nanosecond
regime. This inherently creates a level of complexity and computational
optimisation required to simulate photonic and electronic networks that
standard electronics does not have to solve.

Electronic and optical simulations need to be synchronised into a single
time domain to have a continuous signal. This means that it is necessary
to integrate electrical and optical solvers in some form and to some
level of reasonable translation. It is not necessary to simulate a whole
system at picosecond resolution to observe photonic transient effects,
but to observe transient effects, picosecond resolution might be desired
- whilst steady state might not.

This leads to a complex proposition: how to integrate transient and
steady state time-domain solvers to simulate both electronics and
photonics systems?

Tools Integration
-----------------

The implementation mechanism followed is to create parametric SPICE
components that can be closely integrated with our existing device
models and have a mapping to the electronic and photonic simulation
software. The way this is implemented also follows microservices
architectures as to minimise the computational architecture required for
these systems.

However, one main aspect of complexity is the event-driven nature of multi-physical simulations. For example, a photonic pulse detection event creates an analog stimuli in time which triggers a digital simulation event. As such, simulating accurately, multi-physical triggers is a tricky implementation problem due to the linking events between the domain solvers.

**Why is this analysis important?** Photonic operations and electronic
operations may not be occurring at the same rate. The optical signals
may be changing when the electronics is steady, the electronic signals
may be changing when the photonics is steady, or the electronics and the
photonics are changing both at the same time. These are the potential
scenarios that time-dependent simulations must account for.

Integration Schemes Discussion \& Philosophy
--------------------------------------------------------------------

To try to build a monolithic simulator that can solve for all of these
domains at the same time limits the potential further multi-physical
modelling of more complex systems, such as quantum states. As such, we
want to be as open and as modular as possible in order to make this
simulation tool as useful as possible.

References on mixed-signal simulation with an open-source focus:

-  `Verilog-AMS Flow <https://www.cadence.com/content/dam/cadence-www/global/en_US/documents/services/ams-methodology-ov.pdf>`__
-  `cocotb examples - Mixed-signal (analog/digital) <https://docs.cocotb.org/en/stable/examples.html#mixed-signal-analog-digital>`__
-  `NGSPICE-XSPICE Integration <https://ngspice.sourceforge.io/xspicehowto.html>`__
-  `ISOTEL NGSPICE Mixed-Signal Flow <https://www.isotel.eu/mixedsim/embedded/motorforce/index.html#building-a-chip-with-isotel-d-process-and-embedded-c-code>`__
-  `SPICE OPUS <https://www.spiceopus.si/documentation.html>`__

Of all the open-source mixed-signal simulator software, ``ngspice`` is in strong active development by a large amount of organisations. It makes sense to bootstrap on top of it for these simulations. However, we need to implement event-driven simulators accordingly with both photonic-event driven inputs.

Implementation Scheme
~~~~~~~~~~~~~~~~~~~~~

``ngspice`` can be treated as a dominant simualator. As of writing, this is based on the latest `version 40 manual <https://ngspice.sourceforge.io/docs/ngspice-manual.pdf>`__ . It is capable of implementing event-driven mixed-mode models as described in Chapter 12. This is fundamentally based on building ``.model`` directives in the SPICE netlist, and compiling ``ngspice`` with the ``--enable-xspice`` in our simulation ``.configure`` command.

.. include:: potential_mixed_signal_schemes.rst
