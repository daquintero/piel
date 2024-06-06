__all__ = [
    "test_basic_open_lane_configuration_v1",
    "test_basic_open_lane_configuration_v2",
    "test_spm_open_lane_configuration",
    "example_open_lane_configuration",
]

test_spm_open_lane_configuration = {
    "PDK": "sky130A",
    "DESIGN_NAME": "spm",
    "VERILOG_FILES": ["./src/spm.v"],
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 10,
}

test_basic_open_lane_configuration_v1 = {
    "PDK": "sky130A",
    "DESIGN_NAME": "top",
    "VERILOG_FILES": "dir::src/*.v",
    "RUN_CTS": False,
    "CLOCK_PORT": None,
    "PL_RANDOM_GLB_PLACEMENT": True,
    "FP_SIZING": "absolute",
    "PL_TARGET_DENSITY": 0.75,
    "FP_PDN_AUTO_ADJUST": False,
    "FP_PDN_VPITCH": 25,
    "FP_PDN_HPITCH": 25,
    "FP_PDN_VOFFSET": 5,
    "FP_PDN_HOFFSET": 5,
    "DIODE_INSERTION_STRATEGY": 3,
    "RUN_LINTER": False,
}

test_basic_open_lane_configuration_v2 = {  # Works for small designs
    "PDK": "sky130A",
    "DESIGN_NAME": "top",
    "VERILOG_FILES": "dir::src/*.v",
    "RUN_CTS": False,
    "CLOCK_PORT": None,
    "FP_SIZING": "absolute",
    "GRT_REPAIR_ANTENNAS": True,
    "FP_CORE_UTIL": 80,
    "RUN_HEURISTIC_DIODE_INSERTION": True,
    "RUN_MCSTA": False,  # Temporary TODO REMOVE
}

example_open_lane_configuration = {
    "CLOCK_PERIOD": 100,
    "CLOCK_NET": "clk_o",
    "CLOCK_PORT": None,
    "CLOCK_TREE_SYNTH": True,
    "DIE_AREA": "0 0 100.0 100.00",
    "DESIGN_NAME": None,
    "FP_CORE_UTIL": 40,
    "FP_PDN_AUTO_ADJUST": 0,
    "FP_PDN_VPITCH": 25,
    "FP_PDN_HPITCH": 25,
    "FP_PDN_VOFFSET": 5,
    "FP_PDN_HOFFSET": 5,
    "FP_PIN_ORDER_CFG": "dir::pin_order.cfg",
    "FP_SIZING": "absolute",
    "PL_TARGET_DENSITY": 0.75,
    "PL_RANDOM_GLB_PLACEMENT": True,
    "SYNTH_NO_FLAT": 1,
    "SYNTH_CAP_LOAD": 100,
    "SYNTH_FLAT_TOP": True,
    "SYNTH_MAX_FANOUT": 10,
    "SYNTH_PARAMETERS": "",
    "VERILOG_FILES": None,
    "VERILOG_INCLUDE_DIRS": "dir::src/",
    "pdk::sky130*": {
        "FP_CORE_UTIL": 45,
        "scl::sky130_fd_sc_hd": {"CLOCK_PERIOD": 10},
        "scl::sky130_fd_sc_hdll": {"CLOCK_PERIOD": 10},
        "scl::sky130_fd_sc_hs": {"CLOCK_PERIOD": 8},
        "scl::sky130_fd_sc_ls": {"CLOCK_PERIOD": 10, "SYNTH_MAX_FANOUT": 5},
        "scl::sky130_fd_sc_ms": {"CLOCK_PERIOD": 10},
    },
    "pdk::gf180mcu*": {
        "CLOCK_PERIOD": 24.0,
        "FP_CORE_UTIL": 40,
        "SYNTH_MAX_FANOUT": 4,
        "PL_TARGET_DENSITY": 0.5,
    },
}
