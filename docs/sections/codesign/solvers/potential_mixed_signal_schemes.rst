Legacy - Considered Potential Implementation Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discretisation of the Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This potential implementation consists of:

-  Modularise the time-domain operations
-  Compute photonic and electronic transients only when they are
   changing.
-  Append the corresponding data into a total global time with various
   levels of resolution depending on the reference signal

The goal is to enable to minimize computational cost of simulating such
a system, whilst being flexible and simple enough for any potential
system simulation configuration. This means that we are not simulating
an electronic and photonic system at a femto-second resolution, but we
still resolve the time-scales throughout signal transitions if we
desire.

The electronic simulation implementation is what could be called
micro-SPICE in the sense that it is a minimal implementation of
transition simulation for a circuit. The ideal is that transients
are computed in high resolution and steady-state is computed
operationally.

However, this scheme is invalid when considering feedback signals.

Assuming Independent Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One potential implementation of operation specific electronic transient
modelling is fundamental. Say, an electronic circuit is idle. This means
for the period of time that this circuit is idle and the environment is
constant, it can be approximated that the circuit is in a very similar
constant state with some level of variation dependent only on external
factors eg. temperature or so on. We can also say that the initial
conditions of the circuit, say some DC biasing,
voltage-controlled-current-sources or current-controlled-voltage-sources
DC, or similar are controlled by signals that may be external to the
circuit in question. Now, this is valid whenever there is no remaining
transient memory effects on the initial condition of the circuit, in
this case you need to model full transient effects as in normal SPICE.

Say, our electronic circuit is initially in idle state (which we control
the external parameters that affect this state), and we can control
initial conditions such as the DC biasing and other time-independent
states of our circuit. An opto-electronic signal creates a
time-dependent transient input signal, and we begin our analogue
simulation of our circuit, the pulse has an end width, and the
time-dependent input signal reduces to back to a zero or steady noisy
time-independent input. The internal time-dependent properties of the
circuit reset to their time-independent steady-state.

Then there is the other aspect of complexity, which is how to enable the
multi-domains to access the other domain simulation data state.

I know some experienced people who are reading this are probably
thinking: why bother even doing this, surely tools such as Cadence
already solve this through their mixed-signals AMS simulators. However,
any extension of this is strongly limited by their often incomplete
documentation and closed-source software. It is also limited by
excluding people who do not know Verilog-A (even if I do), such as many
photonics engineers who design their chips in Python. This is some
motivation behind a ``piel`` design flow.

One way is simply not allow them to access any other domain than their
own.


Motivation for Integrated Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the digital time-step, there is a possibility that an analogue signal
contains memory or previous states from the transitions. This creates an
aspect of complexity in modelling these systems. It means that the
time-synchronisation between the digital and analogue system must be
possible. The aspect of complexity is the multi-domain interaction
between the solvers, for example, if the digital solver is triggered by
a photonic-analogue signal or related.


Co-Routine Dominant Solver Limits Feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, this is possible by implementing subroutines. In ``cocotb``,
the simulations are run as asynchronous coroutines with an `analog python model <https://docs.cocotb.org/en/stable/analog_model.html>`__. We can follow this
exact principle in terms of implementing a multi-domain simulatior. Each
``cocotb`` simulation has a ``Timer``, and it is possible that for every
step in time that we desire, we can implement another subroutine. This
would be easy to do if the photonic and analogue simulations were
*time-independent* because then the digital signals control the
operation flow, and it can just be considered a functional-based system
dominated by the digital ``Timer``. it is not particularly complicated
either to have feedback in between digital-photonic logic as through
that subroutine, we have full control over the triggering of digital
signals and related. However, my question of complexity really returns
to the analogue interaction of the signals. If it is digitally-directed,
and there is no analogue feedback interaction onto the digital logic,
then the co-routine would not have to change and is not affected other
than driving the analogue signals.

Part of the question becomes on having to run an analogue simulation at
the same time as the digital one in order to verify that there would not
be remnant analogue memory in between the driving pulses. This makes
sense in particular when the digital ``Timer`` clock overlaps in between
the analogue signal as this changes the initial state of the analogue
simulation. We could keep a track of the initial state, but it might be
a bit complex even when considering a standard RC network and driving
pulses accordingly. The other way to do so, is to perform the SPICE
simulation, and discretize at particular points in time as is the way
with AMS.


Photonic Time Delay Synchronisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another complexity of simulating these systems is that photonic pulses
might also be propagating in time alongside the electronic signals.
``sax`` already implements some functionality to analyse the time-delay
of such photonic circuits, but a resolution mechanism is required to
interconnect these circuits together and a corresponding time-delay
model needs to be provided to the components.
