# # Digital Design & Simulation Flow

# There are many tools to perform a digital design flow. I have summarised some of the most relevant ones in the TODO LINK INTEGRATION PAGE. In `piel`, there are a few digital design flow functionalities integrated. However, because the output or interconnections have a set of accepted common types, it is possible to use different design flows to integrate with other tools in the flow.
#
# We will explore two popular ones:
# * `amaranth` aims to be more than just Python-to-HDL design, but a full digital-flow package design tool.
# * `cocotb` is mainly used for writing testbenches in Python and verification of logic.

import amaranth as am
import piel
import simple_design

# ## `amaranth` Design FLow

# Let's design a basic digital module from their example which we can use later. `amaranth` is one of the great Python digital design flows. I encourage you to check out their documentation for basic examples. We will explore a particular `codesign` use case: creating bespoke logic for detector optical signals.
#
# Also, another aspect is that `amaranth` follow a very declarative class-based design flow, which can be tricky to connect with other tools, so `piel` provides some construction functions that are more easily integrated in the current design flow.
#
# Say we have a table of input optical detector click signals that we want to route some phases to map to some outputs and we want to create a mapping. For now, let's consider a basic arbitrary truth table in this form:
#
# |Input Detector Signals | Output Phase Map|
# |-----------------------|-----------------|
# | 00 | 00 |
# | 01 | 10 |
# | 10 | 11 |
# | 11 | 11 |
#
# `piel` provides some easy functions to perform this convertibility. Say, we provide this information as a dictionary where the keys are the names of our input and output signals. This is a similar principle if you have a detector-triggered DAC bit configuration too.

detector_phase_truth_table = {
    "detector_in": ["00", "01", "10", "11"],
    "phase_map_out": ["00", "10", "11", "11"],
}


def construct_amaranth_module_from_truth_table(
    truth_table: dict,
    inputs: list[str],
    outputs: list[str],
    combinatorial: bool = True,
):
    """
    Note that in some form in amaranth each statement is a form of construction.
    """

    class TruthTable(am.Elaboratable):
        def __init__(self, truth_table: dict, inputs: list, outputs: list):
            # Initialise all the signals accordingly.
            for key, _ in truth_table.items():
                # Determine signal type or largest width from the values.
                setattr(self, key, am.Signal(2))

        def elaborate(self, platform):
            m = am.Module()
            # We need to iterate over the length of the truth table arrays for the input and output keys.
            # TODO implement multiinput.
            # TODO implement some verification that the arrays are of the same length.
            for i in range(4):
                # We iterate over the truth table values
                with m.If(
                    getattr(self, inputs[0]) == int(truth_table[inputs[0]][i], 2)
                ):
                    # Implements a particular output.
                    # m.d.comb += getattr(getattr(self, outputs[0]), "eq")
                    output_value_i = getattr(getattr(self, outputs[0]), "eq")
                    m.d.comb += output_value_i(
                        int(truth_table[outputs[0]][i], 2)
                    )  # NOQA B009

            return m
            # m.d.comb += setattr(output_attribute_i, "eq", int(truth_table[outputs[0]][i], 2))

    return TruthTable(truth_table, inputs, outputs)


our_truth_table_module = construct_amaranth_module_from_truth_table(
    truth_table=detector_phase_truth_table,
    inputs=["detector_in"],
    outputs=["phase_map_out"],
)

elaborated_module = our_truth_table_module.elaborate(platform=None)

from amaranth.back import verilog

with open("up_counter.v", "w") as f:
    f.write(
        verilog.convert(
            our_truth_table_module,
            ports=[
                our_truth_table_module.detector_in,
                our_truth_table_module.phase_map_out,
            ],
        )
    )


class UpCounter(am.Elaboratable):
    """
    A 16-bit up counter with a fixed limit.

    Parameters
    ----------
    limit : int
        The value at which the counter overflows.

    Attributes
    ----------
    en : Signal, in
        The counter is incremented if ``en`` is asserted, and retains
        its value otherwise.
    ovf : Signal, out
        ``ovf`` is asserted when the counter reaches its limit.
    """

    def __init__(self, limit):
        self.limit = limit

        # Ports
        self.en = am.Signal()
        self.ovf = am.Signal()

        # State
        self.count = am.Signal(16)

    def elaborate(self, platform):
        m = am.Module()

        m.d.comb += self.ovf.eq(self.count == self.limit)

        with m.If(self.en):
            with m.If(self.ovf):
                m.d.sync += self.count.eq(0)
            with m.Else():
                m.d.sync += self.count.eq(self.count + 1)
        return m


# `amaranth` is much easier to use than other design flows like `cocotb` because it can be purely interacted with in `Python`, which means there are fewer complexities of integration. However, if you desire to use this with other digital layout tools, for example, OpenROAD as we have previously seen and maintain a coherent project structure with the photonics design flow, `piel` provides some helper functions to achieve this easily.

# +
from amaranth.sim import Simulator

dut = UpCounter(25)


def bench():
    # Disabled counter should not overflow.
    yield dut.en.eq(0)
    for _ in range(30):
        yield
        assert not (yield dut.ovf)

    # Once enabled, the counter should overflow in 25 cycles.
    yield dut.en.eq(1)
    for _ in range(25):
        yield
        assert not (yield dut.ovf)
    yield
    assert (yield dut.ovf)

    # The overflow should clear in one cycle.
    yield
    assert not (yield dut.ovf)


sim = Simulator(dut)
sim.add_clock(1e-6)  # 1 MHz
sim.add_sync_process(bench)
with sim.write_vcd("up_counter.vcd"):
    sim.run()
# -

# ## `cocoTb` Simulation

# Location of our output files

design_directory = piel.return_path(simple_design)
simulation_output_files_directory = piel.return_path(simple_design) / "tb" / "out"
simulation_output_files_directory

simulation_output_files_directory.exists()

# Check that there exist cocotb python test files already in our design directory:

piel.check_cocotb_testbench_exists(simple_design)

# Create a cocotb Makefile

piel.configure_cocotb_simulation(
    design_directory=simple_design,
    simulator="icarus",
    top_level_language="verilog",
    top_level_verilog_module="adder",
    test_python_module="test_adder",
    design_sources_list=list((design_directory / "src").iterdir()),
)

# Now we can create the simulation output files from the `makefile`. Note this will only work in our configured Linux environment.

# Run cocotb simulation
piel.run_cocotb_simulation(design_directory)

# However, what we would like to do is extract timing information of the circuit in Python and get corresponding
# graphs. We would like to have this digital signal information interact with our photonics model. Note that when
# running a `cocotb` simulation, this is done through asynchronous coroutines, so it is within the testbench file
# that any interaction and modelling with the photonics networks is implemented.

# ## Data Visualisation

# The user is free to write their own data visualisation structure and toolset. Conveniently, `piel` does provide a standard plotting tool for these type of `cocotb` signal outputs accordingly.
#
# We first list the files for our design directory, which is out `simple_design` local package if you have followed the correct installation instructions, and we can input our module import rather than an actual path unless you desire to customise.

cocotb_simulation_output_files = piel.get_simulation_output_files_from_design(
    simple_design
)
cocotb_simulation_output_files

# ```python
# ['C:\\Users\\dario\\Documents\\phd\\piel\\docs\\examples\\designs\\simple_design\\simple_design\\tb\\out\\adder_randomised_test.csv']
# ```

# We can read the simulation output data accordingly:

example_simple_simulation_data = piel.read_simulation_data(
    cocotb_simulation_output_files[0]
)
example_simple_simulation_data

# Now we can plot the corresponding data using the built-in interactive `bokeh` signal analyser function:

piel.simple_plot_simulation_data(example_simple_simulation_data)

# This looks like this:

# ![example_simple_design_outputs](../_static/img/examples/02_cocotb_simulation/example_simple_design_outputs.PNG)
