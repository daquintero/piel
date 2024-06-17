#!/bin/bash
# Makefile
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES += /home/daquintero/phd/piel/docs/examples/07_full_flow_demo_electronic_photonic/full_flow_demo/full_flow_demo/src/truth_table_module.v
VERILOG_SOURCES += /home/daquintero/phd/piel/docs/examples/07_full_flow_demo_electronic_photonic/full_flow_demo/full_flow_demo/src/.ipynb_checkpoints
TOPLEVEL := top
MODULE := test_top
include $(shell cocotb-config --makefiles)/Makefile.sim
