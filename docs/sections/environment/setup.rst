Environment Requirements
=========================

Many of the great tools for microelectronic design are functional only
through Linux environments. This leads to an inherent barrier-to-entry
for designers that have not used Linux before. However, it is important
to note that not *all* tools require a Linux flow, and it is possible to
do co-design at different stages in different environments. This could
be useful to distributed teams, or at different stages of a design flow between prototyping a design and full optimisation iterations.

To guarantee full functionality of ``piel``, the recommended environment installation is following the standard IIC-OSIC-TOOLS installation configuration. However, this table of tools compatibility is provided should you use a specific set of toolsets in a given environment. No functionality guarantees are provided in that case.


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
      * - ``pyspice``
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

Docker Configuration
--------------------

.. toctree::
    docker_setup
    docker_environment_configuration
    relevant_docker_commands
    developer_docker_configuration
