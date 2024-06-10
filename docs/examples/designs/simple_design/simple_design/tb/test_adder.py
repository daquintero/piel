# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0
# Simple tests for an adder module
import random
import cocotb
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time
import pandas as pd

if cocotb.simulator.is_running():
    from simple_design.models import adder_model


@cocotb.test()
async def adder_basic_test(dut):
    """Test for 5 + 10"""

    A = 5
    B = 10

    dut.A.value = A
    dut.B.value = B

    await Timer(2, units="ns")

    assert dut.X.value == adder_model(
        A, B
    ), f"Adder result is incorrect: {dut.X.value} != 15"


@cocotb.test()
async def adder_randomised_test(dut):
    """Test for adding 2 random numbers multiple times"""
    simulation_data = dict()
    a_signal_data = list()
    b_signal_data = list()
    x_signal_data = list()
    time_data = list()

    print("Example dut.X.value Print")

    a_signal_data.append(dut.A.value)
    b_signal_data.append(dut.B.value)
    x_signal_data.append(dut.X.value)
    time_data.append(get_sim_time())

    for i in range(10):  # NOQA: B007
        A = random.randint(0, 15)
        B = random.randint(0, 15)

        dut.A.value = A
        dut.B.value = B

        await Timer(2, units="ns")

        print(dut.X.value)

        a_signal_data.append(dut.A.value)
        b_signal_data.append(dut.B.value)
        x_signal_data.append(dut.X.value)
        time_data.append(get_sim_time())

        assert dut.X.value == adder_model(
            A, B
        ), "Randomised test failed with: {A} + {B} = {X}".format(
            A=dut.A.value, B=dut.B.value, X=dut.X.value
        )

    simulation_data = {
        "a": a_signal_data,
        "b": b_signal_data,
        "x": x_signal_data,
        "t": time_data,
    }

    pd.DataFrame(simulation_data).to_csv("out/adder_randomised_test.csv")
