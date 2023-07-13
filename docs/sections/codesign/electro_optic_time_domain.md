# Electro-Optic Time Domain

Electronic and optical simulations need to be synchronised into a single time domain to have a continuous signal. This means that it is necessary to integrate electrical and optical solvers in some form and to some level of reasonable translation. It is not necessary to simulate a whole system at picosecond resolution to observe photonic transient effects, but to observe transient effects, picosecond resolution might be desired - whilst steady state might not.

This leads to a complex proposition: how to integrate transient and steady state time-domain solvers to simulate both electronics and photonics systems? `piel` solves this in a particular way: transients are computed in high resolution and steady-state is computed operationally.

However, this leads to another further complexity, how to integrate analog electronic solvers that might represent our individually connected devices and interconnect, in relation to the rest of the electronic driving system. For example, different heater phase shifters might have different resistances, and although the ports might be the same, they might also have a different interconnect.

## Tools Integration

The implementation mechanism followed is to create parametric SPICE components that can be closely integrated with our existing device models and have a mapping to the electronic and photonic simulation software. The way this is implemented also follows microservices architectures as to minimise the computational architecture required for these systems.

## Photonic Time Delay Synchronisation

Another complexity of simulating these systems is that photonic pulses might also be propagating in time alongside the electronic signals. `sax` already implements some functionality to analyse the time-delay of such photonic circuits, but a resolution mechanism is required to interconnect these circuits together and a corresponding time-delay model needs to be provided to the components.
