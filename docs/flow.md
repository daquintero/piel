# Flow

The flow that `piel` follows depends on the implementation platoform:

## Silicon
1. A digital design
   1. Custom HDL provided by user
2. `OpenROAD` silicon layout using a PDK and performance data
   1. `caravel` integration.
3. `porf` performance parameter data extraction
   1. `OpenSTA` timing-data extraction.
4. `piel`-`gdsfactory` implementation onto an integrated interconnection netlist.
5. `sax` system-frequency domain model.
6. `verilator` time-domain digital simulator.

## FPGA
TODO Detail the toolset.

## Implementation Principle

`piel` is written for functional composition as for ease of integration with `GDSFactory`.
