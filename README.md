# `piel` - Photonic and Integrated ELectronic tools
[![PyPI Version](https://img.shields.io/pypi/v/piel.svg)](https://pypi.python.org/pypi/piel)
[![Build Status](https://img.shields.io/travis/daquintero/piel.svg)](https://travis-ci.com/daquintero/piel)
[![Documentation Status](https://readthedocs.org/projects/piel/badge/?version=latest)](https://piel.readthedocs.io/en/latest/?version=latest)
[![Updates](https://pyup.io/repos/github/daquintero/piel/shield.svg)](https://pyup.io/repos/github/daquintero/piel/)

Microservices to codesign photonics, electronics, quantum, and more.

- Free software: MIT license
- Documentation: [https://piel.readthedocs.io](https://piel.readthedocs.io)

## Target functionality
* Co-simulation and optimisation between integrated photonic and electronic chip design.
* System interconnection modelling in multiple environments.
* Individual and interposer design integration.
* Multi-domain electronics and photonics component models.
* Quantum models of physical circuitry

`piel` aims to provide an integrated workflow to co-design photonics and electronics, classically and quantum. It does not aim to replace the individual functionality of each design tool, but rather provide a glue to easily connect them all together and extract the system performance.

## Examples

Follow the many [examples in the documentation](https://piel.readthedocs.io/en/stable/examples.html).

## Microservices Toolset
This package provides interconnection functions to easily co-design microelectronics through the functionality of the [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools) and photonics via [GDSFactory](https://github.com/gdsfactory/gdsfactory).

Some existing microservice dependency integrations are:
* [cocotb](https://github.com/cocotb/cocotb) - a coroutine based cosimulation library for writing VHDL and Verilog testbenches in Python.
* [GDSFactory](https://github.com/gdsfactory/gdsfactory) - An open source platform for end to-end photonic chip design and validation
* [OpenLane v1](https://github.com/The-OpenROAD-Project/OpenLane) - an automated RTL to GDSII flow based on several components including OpenROAD, Yosys, Magic, Netgen and custom methodology scripts for design exploration and optimization
* [pyspice](https://github.com/PySpice-org/PySpice) - Simulate electronic circuit using Python and the Ngspice / Xyce simulators
* [sax](https://github.com/flaport/sax) - S-parameter based frequency domain circuit simulations and optimizations using JAX.
* [qutip](https://github.com/qutip/qutip) - QuTiP: Quantum Toolbox in Python

## Environment Requirements
* Please install the Linux Docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).


## Contribution
If you feel dedicated enough to become a project maintainer, or just want to do a single contributor, let's do this together!
