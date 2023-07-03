# Overview

`piel` allows the integration and co-simulation between electronics and photonics through with `gdsfactory` for a range
of quantum photonic circuits.

## Roadmap

`piel` will be composed of the following modules all which are contained in the docker environment file provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools):
* [`cocotb`](https://github.com/cocotb/cocotb) - implements the methods that allow the configuration of signal
  stimulus to the electronic logic directly from Python.
* [`gdsfactory`]() -
* [`porf`]() -  performance parameter data extraction
* [`OpenLane v1`](https://github.com/The-OpenROAD-Project/OpenLane) - implements the RTL-to-GDSII flow for the
  electronic logic and outputs performance parameters of the implemented circuitry.
* [`OpenSTA`]() - timing-data extraction.
* [`sax`]() system-frequency domain model.
* [`yosys`]() -
* [`verilator`]() - time-domain digital simulator.
