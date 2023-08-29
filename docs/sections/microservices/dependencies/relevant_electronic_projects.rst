Relevant Electronic Projects
============================

This contains a list of important open-source software that tackles some
challenges or individual aspects of electronic or electronic-photonic
codesign.

+----------------------+------------------------------------+----------+
| Name                 | Description                        | Status   |
+======================+====================================+==========+
| `Amaranth <https     | Industry-supported, multi-FPGA     | Ear      |
| ://github.com/amaran | support, larger-scale integration  | ly-Stage |
| th-lang/amaranth>`__ | planned.                           | Active   |
|                      |                                    | Dev      |
|                      |                                    | elopment |
+----------------------+------------------------------------+----------+
| `Magma               | Tries to replicate verilog syntax  | Active   |
| Lang <htt            | in Python. Hard to read.           | Dev      |
| ps://magma.readthedo |                                    | elopment |
| cs.io/en/latest/>`__ |                                    |          |
+----------------------+------------------------------------+----------+
| `MyHDL <https:       | Silicon-proven python-to-silicon.  | Active   |
| //www.myhdl.org/>`__ | Weird LGPL license.                | Dev      |
|                      |                                    | elopment |
+----------------------+------------------------------------+----------+
| `V                   | Interchange formats for chip       | Active   |
| LSIR <https://github | design.                            | Dev      |
| .com/Vlsir/Vlsir>`__ |                                    | elopment |
+----------------------+------------------------------------+----------+

There’s a list of awesome electronics projects `compiled by the CEO of
RapidSilicon <https://github.com/aolofsson/awesome-opensource-hardware>`__.

Selection of Python-to-HDL Flow
-------------------------------

I am currently deciding which project to integrate as part of the
python-to-logic design flow. There are some cases where we might want to
generate some logic from some specific logical functionality. There are
several projects available, but not all with suitable enough
functionality:

+-----------------------+--------------------------------+-------------+
| Name                  | Description                    | Status      |
+=======================+================================+=============+
| `Amaranth <htt        | Industry-supported, multi-FPGA | Early-Stage |
| ps://github.com/amara | support, larger-scale          | Active      |
| nth-lang/amaranth>`__ | integration planned.           | Development |
+-----------------------+--------------------------------+-------------+
| `Magma                | Tries to replicate verilog     | Active      |
| Lang <h               | syntax in Python. Hard to      | Development |
| ttps://magma.readthed | read.                          |             |
| ocs.io/en/latest/>`__ |                                |             |
+-----------------------+--------------------------------+-------------+
| `MyHDL <https         | Silicon-proven                 | Active      |
| ://www.myhdl.org/>`__ | python-to-silicon. Weird LGPL  | Development |
|                       | license.                       |             |
+-----------------------+--------------------------------+-------------+

I think going with Amaranth makes sense.
