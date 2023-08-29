Useful Commands
===============

``cocotb`` is an incredible tool for logical time-dependent digital
systems verification. What we aim to do is extend it to interconnect out
photonic models according to the electronic signals applied. There are a
few useful commands worth noting:

.. list-table:: Verilog Top-Module Device Commands
   :header-rows: 1

   * - Description
     - Command
   * - Get the ``dut``, the Verilog top-module device under test.
     - ``dut``
   * - Get a list of the ``dut`` signals, modules and user-defined parameters accessible through dot notation.
     - ``dir(dut)``
   * - Get a signal value within the top level at a particular subroutine time.
     - ``dut.<"signal">.value``
   * - Get the current simulation time.
     - ``cocotb.utils.get_sim_time()``


One thing we would like to do is directly connect the time-domain
digital simulation data to the inputs of the photonic network
performance. It is possible to access the signal data at any point using
the dot notation and the subroutine desired in ``cocotb``. This is why,
while ``cocotb`` will enable writing and reading ``vcd`` standard
waveform files such as those that can be inputted into ``GTKWave`` and
similar viewers, the main examples of the co-design between electronics
and photonics will not use this scheme. Another microservice could be
implemented should a user desire.

It is desired to save the corresponding data in a much easier to use
format in order to easily plot the waveforms and optical signals through
a common python plotting tool. You can see how to save data into a
pandas DataFrame file in
``docs/examples/designs/simple_design/simple_design/tb/test_adder.py``.
TODO link.
