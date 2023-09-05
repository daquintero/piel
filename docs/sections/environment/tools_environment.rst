Tool Environment Requirements
===============================

Many of the great tools for microelectronic design are functional only
through Linux environments. This leads to an inherent barrier-to-entry
for designers that have not used Linux before. However, it is important
to note that not *all* tools require a Linux flow, and it is possible to
do co-design at different stages in different environments. This could
be useful to distributed teams, or at different stages of a design flow between prototyping a design and full optimisation iterations.

One of the main complexities of multi-tool open source design is the version management of the tools, and making sure they all speak to each other without conflicting tool environment requirements. In ``piel``, we provide a set of environment solutions to guarantee the shared functionality between the toolsets. This means we forcibly tie down each tool version to what we know works and that our examples pass as a result. This means that if you want to upgrade to a latest version, you will need to do the upgrade and verification that the tool compatibility passes.

The implementation of this consists of tying down the requirements of the primary dependencies of the project, but not the secondary ones. This means that we expect the secondary dependencies to resolve through the dependency management system when providing a suitable primary dependency environment.

To guarantee full functionality of ``piel``, we provide some customised environment installations, particularly through ``nix`` in Linux.

Dependency Environment Breakdown
-----------------------------------------------

This table of tools compatibility is provided should you use a specific set of toolsets in a given environment. No complete functionality guarantees are provided in this case.


.. list-table:: Tools Compatibility
      :header-rows: 1

      * - Tool
        - Windows
        - Linux
        - OS X ?
        - Possible Integrations
      * - ``cocotb``
        - ``iverilog``
        - ``iverilog``, ``verilator``, Cadence ``xcelium``
        - ``iverilog``
        - ``iverilog``, ``verilator``, Cadence ``xcelium``
      * - ``gdsfactory``
        - *
        - *
        - *
        - Many
      * - ``openlane``
        -
        - *
        -
        -
      * - ``hdl21``
        - ``ngspice`` & ``xyce``
        - ``ngspice`` & ``xyce``
        - ``ngspice`` & ``xyce``
        - ``ngspice``, ``xyce``
      * - ``sax``
        - *
        - *
        - *
        - *
      * - ``thewalrus``
        - *
        - *
        - *
        - *

This is a preliminary table, I mostly develop on Linux, you need to verify your system configuration or use the recommended Docker environment. TODO I am unsure about the OS X integrations.

In the future, we will have custom installations for different types of users so they can install minimal dependencies for their use case.
