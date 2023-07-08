# SPICE Integration

The implementation mechanism is to provide component models that include the raw interconnect based on similar `gdsfactory` port naming and matching. This will allow us to design netlists that can be closely mapped into a SPICE solver, directly from `gdsfactory`. This may eventually be interconnected through [VLSIR](https://github.com/Vlsir/Vlsir).

Note that for a particular `SPICE` implementation, more than time-domain simulations can be performed as well. This is why `piel` provides functionality to construct simulations to perform these multi-domain simulations and construct these systems from component model primitives.
