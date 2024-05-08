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


Electronic Experimental Setups Reviews
--------------------------------------

The goal of this page is to provide a list of open-source references and toolsets for electronic experimental setups that can be compared to simulation setups.

.. list-table::
   :header-rows: 1

   * - Name
     - Usage
     - Source
   * - General Open-Source Electronics Reviews
     - Available popular open-source general electronics tools
     - https://github.com/ajaymnk/open-electronics
   * - Delay and Dead Time in Integrated MOSFET Drivers
     - Good comparison between simulation and experimental results for PWM setups
     - https://www.ti.com/lit/an/slvaf84/slvaf84.pdf?ts=1715625183585&ref_url=https%253A%252F%252Fwww.google.com%252F
   * - Collection of RF test equipment open-source scripts
     -
     - https://github.com/DavidLutton/LabToolkit
   * - Popular depreciated collection of device controllers
     - A Python implementation of the Interchangeable Virtual Instrument standard.
     - https://github.com/python-ivi/python-ivi
   * - Actively maintained collection of python-based measurement tools
     -
     - https://github.com/pymeasure/pymeasure
   * - A great reference for RF design and simulation
     -
     - https://scikit-rf.readthedocs.io/en/latest/index.html
   * - A generic microwave resource
     -
     - https://www.microwaves101.com/
   * - A generic microwave resource
     -
     - https://www.microwaves101.com/
   * - Keysight Resources for measurement setups
     -
     - https://docs.keysight.com/kkbopen/how-to-synchronize-rf-signal-generators-636240197.html
   * - Keysight Resources for measurement setups
     -
     - https://www.keysight.com/us/en/assets/7018-04627/application-notes/5992-0249.pdf
   * - Modern RF and Microwave Measurement Techniques
     - Really handy explanations of RF and microwave measurement techniques. References are particularly useful.
     - Book
   * - The RF and Microwave Handbook
     - Also pretty handy for testing theory
     - Book
   * -
     -
     -

Pretty handy references:
THis book is the answer to our propagation delay measurement questions: https://frank.pocnet.net/other/sos/Philips_DigitalExersises_AppliedMeasurementsInDigitalAndPulseTechnique.pdf
The testing Chapter 4 of the The RF and Microwave Handbook for microwave stuff https://api.pageplace.de/preview/DT0400.9781439833230_A37606175/preview-9781439833230_A37606175.pdf

