``spice`` & ``pyspice``
=======================

Most photonic components are commonly driven from analogue signals that represent electrical-optical signal conversion. It is necessary to represent these analogue signals before they can be digitised. The analogue electrical load effects need to be considered as well, otherwise, circuits implemented in these tools will inaccurately measure signal transfer between domains.

Another objective of implementing this tool is to enable a clear driver-load-measurement relationship between electronic circuitry and photonic systems. We can integrate `SPICE` solvers with other open source tools to solve photonic computational functions and more.

Part of the objective of `piel` is to be as integrable as possible, and as modular as possible, because the maintainers believe this ensures longer-term usefulness than monoliths.

Integration Scheme
------------------

In `piel`, the `SPICE`-generated circuits will always be raw `SPICE` rather than using a package to create a circuit model or related. This is because any serious `SPICE` program will always accept raw inputs that can be integrated into its solvers. This will allow us in the future to integrate with `SPECTRE` and other related programs.

.. toctree::

    relevant_spice_projects
