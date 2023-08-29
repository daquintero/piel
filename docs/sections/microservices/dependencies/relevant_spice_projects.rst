List of Relevant SPICE Projects
===============================

There are a large amount of open-source SPICE solvers currently
available. In ``piel``, ``hdl21`` has been integrated first because it
is a great piece of software and very simple to install. It is easily
integrated into the IIC-OSIC-TOOLS, which means that if users install
this environment, they will already be capable of using this tools with
this project when installing ``piel``. However, there are other
open-source solvers that may be desired to be integrated, or even
proprietary ones such as Cadence ``spectre`` and more.

We list some here for reference, and further development is welcome to
integrate these in the design flow:

+------------+---------------------+----------------------------------+
| Name       | Description         | Status                           |
+============+=====================+==================================+
| `NGSP      | Open source spice   | Active                           |
| ICE <https | simulator for       |                                  |
| ://ngspice | electric and        |                                  |
| .sourcefor | electronic circuits |                                  |
| ge.io/>`__ |                     |                                  |
+------------+---------------------+----------------------------------+
| `P         | Simulate electronic | Inactive                         |
| YSPICE <ht | circuit using       |                                  |
| tps://gith | Python and the      |                                  |
| ub.com/PyS | Ngspice / Xyce      |                                  |
| pice-org/P | simulators          |                                  |
| ySpice>`__ |                     |                                  |
+------------+---------------------+----------------------------------+
| `XYCE      | Open source,        | Active                           |
| <https://x | SPICE-compatible,   |                                  |
| yce.sandi  | high-performance    |                                  |
| a.gov>`__  | analog circuit      |                                  |
|            | simulator           |                                  |
+------------+---------------------+----------------------------------+
| `QUCS <htt | Quite Universal     | Inactive Mostly, some `PySPICE   |
| ps://githu | Circuit Simulator   | Integration <https://py          |
| b.com/Qucs |                     | spice.fabrice-salvaire.fr/releas |
| /qucs/>`__ |                     | es/v1.5/simulators.html#qucs>`__ |
+------------+---------------------+----------------------------------+

``hdl21`` Selection
-------------------

Note that PYSPICE despite being recently inactive, already has working
bindings to ``ngspice`` and ``XYCE``, which one would think makes it
suitable for python-based integration, rather than attempting to write
this on our own.

The benefit of integrating ``pyspice`` is that it already has a range of
built-in functionality for circuit and signal analysis, that can be
easily ported into ``piel`` and integrated into the codesign process.

Note that ``pyspice`` is a a OOP-based project, which means that
class-based functionality is prevalent when implementing simulations and
systems, these are fundamentally run within the integration functions.

However, in practice, after an initial attempt to integrate PySPICE, it
is evident how the lack of support of the project makes it difficult to
integrate it into the ``piel`` project. The class typing is very archaic
and it was written before ``pydantic`` data validation was a thing.
Also, even if we wanted to integrate new functionality, it is likely the
project would not be fully supported. It is because of this that
``hdl21`` has been selected, as it is a well structured project with
easy integrations and will be part of the ``VLSIR`` ecosystem.
