# `piel` - Photonic and Integrated ELectronic system design
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

`piel` aims to provide an integrated workflow to co-design photonics and electronics. It does not aim to replace the individual functionality of each design tool, but rather provide a glue to easily connect them all together and extract the system performance.

## Dependency Toolset
This package provides a wrapper to easily co-design microelectronics through the functionality of the [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools) and photonics via [GDSFactory](https://github.com/gdsfactory/gdsfactory).

Some individual tools already integrated are:
* [cocotb](https://github.com/cocotb/cocotb) - implements the methods that allow the testbenching configuration of signal stimulus to the electronic logic directly from Python.
* [OpenLane v1](https://github.com/The-OpenROAD-Project/OpenLane) - implements the RTL-to-GDSII flow for the
  electronic logic and outputs performance parameters of the implemented circuitry.

Coming next GDSFactory netlisting and layout integration.

## Environment Requirements
* Please install the Linux Docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).

## Credits
This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- Cookiecutter: [https://github.com/audreyr/cookiecutter](https://github.com/audreyr/cookiecutter)
- `audreyr/cookiecutter-pypackage`: [https://github.com/audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
