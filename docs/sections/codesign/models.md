# Component Models

Multiple physical domains need to be understood of a system to create an optimal interconnection implementation. We provide a set of component models in multiple domains, that can be easily interfaced with each corresponding domain simulator.

Models are provided in the following types:

| Type      | Description                                                                                      | Examples                                                     |
|-----------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| frequency | Frequency domain relating inputs and outputs.                                                    |                                                              |
| logical   | Deterministic relationships from a defined input to an output that implement a logical operation. |                                                              |
| physical  | Relationships between physical parameters and other physical performance parameters.             | Waveguide width vs optical attenuation per meter vs TE mode. |
| time      | Transient domain models of time-dependent state changes.                                         |                                                              |
