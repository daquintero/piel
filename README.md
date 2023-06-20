# `piel` - Photonic-Electronic Simulation and System Design
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

## Dependency Toolset
This package provides a wrapper to easily co-design microelectronics through the functionality of the [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools) and photonics via [GDSFactory](gdsfactory/gdsfactory: An open source platform for end to-end photonic chip design and validation)

* `gdsfactory` for the photonic design
* `OpenROAD OpenLane` for the micro-electronic layout design
* `verilator` & `iverilog` for the digital HDL simulations
* `cocotb` for python-based testbench modelling
* `porf` my custom package for `OpenROAD` data extraction.
* [Future] FPGA modelling and integration

## Environment Requirements
* Please install the Linux Docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).

## Credits
This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- Cookiecutter: [https://github.com/audreyr/cookiecutter](https://github.com/audreyr/cookiecutter)
- `audreyr/cookiecutter-pypackage`: [https://github.com/audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
