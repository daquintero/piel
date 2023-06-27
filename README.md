# `piel` - Photonic and Integrated ELectronic tools
[![PyPI Version](https://img.shields.io/pypi/v/piel.svg)](https://pypi.python.org/pypi/piel)
[![Build Status](https://img.shields.io/travis/daquintero/piel.svg)](https://travis-ci.com/daquintero/piel)
[![Documentation Status](https://readthedocs.org/projects/piel/badge/?version=latest)](https://piel.readthedocs.io/en/latest/?version=latest)
[![Updates](https://pyup.io/repos/github/daquintero/piel/shield.svg)](https://pyup.io/repos/github/daquintero/piel/)

Photonic and electronic co-simulation system design tools interfaced with open-source design software like GDSFactory and OpenROAD.

- Free software: MIT license
- Documentation: [https://piel.readthedocs.io](https://piel.readthedocs.io)

## Target functionality
* Co-simulation and optimisation between integrated photonic and electronic chip design.
* System interconnection modelling in multiple environments.
* Individual and interposer design integration.
* Multi-domain electronics and photonics component models

`piel` aims to provide an integrated workflow to co-design photonics and electronics. It does not aim to replace the individual functionality of each design tool, but rather provide a glue to easily connect them all together and extract the system performance.

## Dependency Toolset
This package provides a wrapper to easily co-design microelectronics through the functionality of the [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools) and photonics via [GDSFactory](https://github.com/gdsfactory/gdsfactory).

Some individual tools already integrated are:
* [cocotb](https://github.com/cocotb/cocotb) - a coroutine based cosimulation library for writing VHDL and Verilog testbenches in Python.
* [GDSFactory](https://github.com/gdsfactory/gdsfactory) - An open source platform for end to-end photonic chip design and validation
* [OpenLane v1](https://github.com/The-OpenROAD-Project/OpenLane) - an automated RTL to GDSII flow based on several components including OpenROAD, Yosys, Magic, Netgen and custom methodology scripts for design exploration and optimization
* [sax](https://github.com/flaport/sax) - S-parameter based frequency domain circuit simulations and optimizations using JAX.

## Environment Requirements
* Please install the Linux Docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).
