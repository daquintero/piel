#!/bin/bash
# Makefile
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES += /foss/designs/piel/docs/examples/simple_design/src/adder.vhdl
VERILOG_SOURCES += /foss/designs/piel/docs/examples/simple_design/src/adder.sv
TOPLEVEL := adder
MODULE := test_adder
include $(shell cocotb-config --makefiles)/Makefile.sim
