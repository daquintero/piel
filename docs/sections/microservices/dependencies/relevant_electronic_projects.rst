Relevant Electronic Projects
--------------------------------------------

This contains a list of important open-source software that tackles some
challenges or individual aspects of electronic or electronic-photonic
codesign.

.. list-table:: FPGA and Chip Design Languages
   :header-rows: 1

   * - Name
     - Description
     - Status
   * - `Amaranth <https://github.com/amaranth-lang/amaranth>`__
     - Industry-supported, multi-FPGA support, larger-scale integration planned.
     - Early-Stage Active Development
   * - `Magma Lang <https://magma.readthedocs.io/en/latest/>`__
     - Tries to replicate Verilog syntax in Python. Hard to read.
     - Active Development
   * - `MyHDL <https://www.myhdl.org/>`__
     - Silicon-proven python-to-silicon. Weird LGPL license.
     - Active Development
   * - `VLSIR <https://github.com/Vlsir/Vlsir>`__
     - Interchange formats for chip design.
     - Active Development


There’s a list of awesome electronics projects `compiled by the CEO of
RapidSilicon <https://github.com/aolofsson/awesome-opensource-hardware>`__.

Selection of Python-to-HDL Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

I am currently deciding which project to integrate as part of the
python-to-logic design flow. There are some cases where we might want to
generate some logic from some specific logical functionality. There are
several projects available, but not all with suitable enough
functionality:

.. list-table:: FPGA and Chip Design Languages
   :header-rows: 1

   * - Name
     - Description
     - Status
   * - `Amaranth <https://github.com/amaranth-lang/amaranth>`__
     - Industry-supported, multi-FPGA support, larger-scale integration planned.
     - Early-Stage Active Development
   * - `Magma Lang <https://magma.readthedocs.io/en/latest/>`__
     - Tries to replicate Verilog syntax in Python. Hard to read.
     - Active Development
   * - `MyHDL <https://www.myhdl.org/>`__
     - Silicon-proven python-to-silicon. Weird LGPL license.
     - Active Development


I think going with Amaranth makes sense.
