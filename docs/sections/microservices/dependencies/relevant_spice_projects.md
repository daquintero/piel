# List of Relevant SPICE Projects

There are a large amount of open-source SPICE solvers currently available. In `piel`, `pyspice` has been integrated first because it is part of the `IIC-OSIC-TOOLS` ecosystem, which means that if users install this environment, they will already be capable of using this tools with this project out of the box. However, there are other open-source solvers that may be desired to be integrated, or even proprietary ones such as Cadence `spectre` and more.

We list some here for reference, and further development is welcome to integrate these in the design flow:


| Name                                       | Description                                                              | Status                                                                                                              |
|--------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [NGSPICE](https://ngspice.sourceforge.io/) | Open source spice simulator for electric and electronic circuits         | Active                                                                                                              |
| [PYSPICE](https://github.com/PySpice-org/PySpice) | Simulate electronic circuit using Python and the Ngspice / Xyce simulators         | Recently Inactive                                                                                                   |
| [XYCE](https://xyce.sandia.gov/)           | Open source, SPICE-compatible, high-performance analog circuit simulator | Active                                                                                                              |
| [QUCS](https://github.com/Qucs/qucs/)      | Quite Universal Circuit Simulator                                        | Inactive Mostly, some [PySPICE Integration](https://pyspice.fabrice-salvaire.fr/releases/v1.5/simulators.html#qucs) |

## `pyspice` selection

Note that PYSPICE despite being recently inactive, already has working bindings to `ngspice` and `XYCE`, which makes it suitable for python-based integration, rather than attempting to write this on our own.

The benefit of integrating `pyspice` is that it already has a range of built-in functionality for circuit and signal analysis, that can be easily ported into `piel` and integrated into the codesign process.

Note that `pyspice` is a a OOP-based project, which means that class-based functionality is prevalent when implementing simulations and systems, these are fundamentally run within the integration functions.
