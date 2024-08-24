from openlane.flows import SequentialFlow
from openlane.steps import Yosys, OpenROAD, Magic, Netgen


class MyFlow(SequentialFlow):
    Steps = [
        Yosys.Synthesis,
        OpenROAD.Floorplan,
        OpenROAD.TapEndcapInsertion,
        OpenROAD.GeneratePDN,
        OpenROAD.IOPlacement,
        OpenROAD.GlobalPlacement,
        OpenROAD.DetailedPlacement,
        OpenROAD.GlobalRouting,
        OpenROAD.DetailedRouting,
        OpenROAD.FillInsertion,
        Magic.StreamOut,
        Magic.DRC,
        Magic.SpiceExtraction,
        Netgen.LVS,
    ]


flow = MyFlow(
    {
        "PDK": "sky130A",
        "DESIGN_NAME": "top",
        "VERILOG_FILES": [
            "/home/daquintero/phd/piel/docs/examples/designs/amaranth_driven_flow/amaranth_driven_flow/src/truth_table_module.v"
        ],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": 10,
    },
    design_dir=".",
)
flow.start()
