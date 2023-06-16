# About

`piel` allows the integration and co-simulation between electronics and photonics through with `gdsfactory` for a range
of quantum photonic circuits.

# Project Structure

`piel` is composed of the following modules all which are contained in the docker environment file provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools):
* [`caravel`]() -
* [`cocotb`](https://github.com/cocotb/cocotb) - implements the methods that allow the configuration of signal
  stimulus to the electronic logic directly from Python.
* [`gdsfactory`]() -
* [`porf`]() -
* [`OpenLane`](https://github.com/The-OpenROAD-Project/OpenLane) - implements the RTL-to-GDSII flow for the
  electronic logic and outputs performance parameters of the implemented circuitry.
* [`OpenSTA`]() -
* [`yosys`]() -
* [`verilator`]() -
