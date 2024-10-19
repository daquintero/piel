import pytest
import piel
import pandas as pd
import numpy as np
import sax
from piel.models.physical.photonic import (
    mzi2x2_2x2_phase_shifter,
    straight_heater_metal_simple,
)
from piel.models.physical.electronic import get_default_models
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.export import to_svg
from piel.tools.hdl21 import (
    configure_operating_point_simulation,
    configure_transient_simulation,
    run_simulation,
)
from piel.visual import plot_simple, plot_simple_multi_row
from piel.models.physical.electro_optic import linear_phase_mapping_relationship
import sys
import hdl21 as h


# Activate generic PDK for tests
@pytest.fixture(scope="module", autouse=True)
def activate_pdk():
    get_generic_pdk().activate()


def test_netlist_extraction():
    """Test extraction of the electrical netlist from the phase shifter."""
    phase_shifter = mzi2x2_2x2_phase_shifter()
    netlist = phase_shifter.get_netlist(exclude_port_types="optical")
    assert "instances" in netlist
    assert "sxt" in netlist["instances"]
    sxt_instance = netlist["instances"]["sxt"]
    assert sxt_instance["info"]["resistance"] is not None
    assert isinstance(sxt_instance["info"]["resistance"], float)


def test_netlist_flat_keys():
    """Test keys of the flattened netlist."""
    phase_shifter = mzi2x2_2x2_phase_shifter()
    netlist_flat = phase_shifter.get_netlist_flat(exclude_port_types="optical")
    assert "connections" in netlist_flat
    assert isinstance(netlist_flat["connections"], dict)


def test_resistive_phase_shifter():
    """Test the resistive phase shifter instance and its resistance."""
    phase_shifter = mzi2x2_2x2_phase_shifter()
    netlist = phase_shifter.get_netlist(exclude_port_types="optical")
    sxt_info = netlist["instances"]["sxt"]["info"]
    assert sxt_info["resistance"] == 0  # Default resistance
    # Modify resistance
    phase_shifter_netlist = phase_shifter.get_netlist(exclude_port_types="optical")
    phase_shifter_netlist["instances"]["sxt"]["info"]["resistance"] = 1000.0
    assert phase_shifter_netlist["instances"]["sxt"]["info"]["resistance"] == 1000.0


def test_short_resistive_phase_shifter():
    """Test a variation of the phase shifter with modified length."""
    short_phase_shifter = mzi2x2_2x2_phase_shifter(length_x=100)
    netlist = short_phase_shifter.get_netlist(exclude_port_types="optical")
    sxt_info = netlist["instances"]["sxt"]["info"]
    assert sxt_info["resistance"] == 0  # As per example
    # Check length modification
    assert short_phase_shifter.settings["length_x"] == 100


def test_svg_export(tmp_path):
    """Test exporting the phase shifter to SVG."""
    phase_shifter = mzi2x2_2x2_phase_shifter()
    svg_path = tmp_path / "phase_shifter.svg"
    to_svg(phase_shifter, filepath=svg_path)
    assert svg_path.exists()


def test_spice_netlist_creation():
    """Test creation of the SPICE netlist from the resistive heater."""
    heater = straight_heater_metal_simple()
    netlist = heater.get_netlist(allow_multiple=True, exclude_port_types="optical")
    spice_netlist = piel.integration.gdsfactory_netlist_with_hdl21_generators(netlist)
    assert "instances" in spice_netlist
    assert "straight_1" in spice_netlist["instances"]


def test_hdl21_module_construction():
    """Test construction of the HDL21 module from the SPICE netlist."""
    heater = straight_heater_metal_simple()
    netlist = heater.get_netlist(allow_multiple=True, exclude_port_types="optical")
    spice_netlist = piel.integration.gdsfactory_netlist_with_hdl21_generators(netlist)
    circuit = piel.integration.construct_hdl21_module(spice_netlist)
    assert isinstance(circuit, h.Module)
    assert "straight_1" in circuit.instances
    assert "e1" in circuit.ports


def test_subcircuit_netlist():
    """Test the netlist of a subcircuit in HDL21."""
    heater = straight_heater_metal_simple()
    netlist = heater.get_netlist(allow_multiple=True, exclude_port_types="optical")
    spice_netlist = piel.integration.gdsfactory_netlist_with_hdl21_generators(netlist)
    circuit = piel.integration.construct_hdl21_module(spice_netlist)
    straight_instance = circuit.instances["straight_1"]
    sub_netlist = h.netlist(straight_instance.of, sys.stdout, fmt="spice")
    assert ".SUBCKT Straight" in sub_netlist


def test_spice_netlist_export(tmp_path):
    """Test exporting the complete SPICE netlist."""
    heater = straight_heater_metal_simple()
    netlist = heater.get_netlist(allow_multiple=True, exclude_port_types="optical")
    spice_netlist = piel.integration.gdsfactory_netlist_with_hdl21_generators(netlist)
    circuit = piel.integration.construct_hdl21_module(spice_netlist)
    spice_file = tmp_path / "heater_circuit.spice"
    with open(spice_file, "w") as f:
        h.netlist(circuit, f, fmt="spice")
    assert spice_file.exists()
    with open(spice_file, "r") as f:
        content = f.read()
    assert ".SUBCKT" in content


def test_operating_point_simulation():
    """Test running a DC operating point simulation."""

    @h.module
    class OperatingPointTb:
        VSS = h.Port()
        VDC = h.Vdc(dc=1)(n=VSS)
        dut = straight_heater_metal_simple()
        dut.e1 = VDC.p
        dut.e2 = VSS

    simulation = configure_operating_point_simulation(
        testbench=OperatingPointTb, name="op_simulation"
    )
    result = run_simulation(simulation)
    assert len(result.an) == 1
    op_result = result.an[0]
    assert "v(xtop.vvdc_p)" in op_result.data
    assert "i(v.xtop.vvdc)" in op_result.data
    # Check expected values (assuming resistance=1000 Ohms)
    assert np.isclose(op_result.data["v(xtop.vvdc_p)"], 1.0)
    assert np.isclose(op_result.data["i(v.xtop.vvdc)"], 0.001)


def test_transient_simulation(tmp_path):
    """Test running a transient simulation and processing results."""

    @h.module
    class TransientTb:
        VSS = h.Port()
        VPULSE = h.Vpulse(
            delay=1 * h.prefix.m,
            v1=-1000 * h.prefix.m,
            v2=1000 * h.prefix.m,
            period=100 * h.prefix.m,
            rise=10 * h.prefix.m,
            fall=10 * h.prefix.m,
            width=75 * h.prefix.m,
        )(n=VSS)
        dut = straight_heater_metal_simple()
        dut.e1 = VPULSE.p
        dut.e2 = VSS

    simulation = configure_transient_simulation(
        testbench=TransientTb,
        stop_time_s=0.2,
        step_time_s=1e-4,
        name="transient_simulation",
    )
    run_simulation(simulation, to_csv=True)
    csv_file = tmp_path / "TransientTb.csv"
    assert csv_file.exists()
    results = pd.read_csv(csv_file)
    assert "time" in results.columns
    assert "v(xtop.vpulse_p)" in results.columns
    assert "i(v.xtop.vvpulse)" in results.columns
    # Simple check on data length
    assert len(results) > 0


def test_phase_mapping_relationship():
    """Test the linear phase mapping function."""
    phase_map = linear_phase_mapping_relationship(
        phase_power_slope=10, zero_power_phase=1
    )
    assert phase_map(0) == 1
    assert phase_map(0.5) == 6
    assert phase_map(1) == 11


def test_energy_consumption_calculation():
    """Test the calculation of power and energy consumption from transient simulation."""
    # Create sample data
    data = {
        "time": np.linspace(0, 0.002, 21),
        "v(xtop.vpulse_p)": np.linspace(0, 1, 21),
        "i(v.xtop.vvpulse)": np.linspace(0, 0.001, 21),
    }
    df = pd.DataFrame(data)
    df["power(xtop.vpulse)"] = df["v(xtop.vpulse_p)"] * df["i(v.xtop.vvpulse)"]
    df["resistance(xtop.vpulse)"] = np.round(
        df["v(xtop.vpulse_p)"] / df["i(v.xtop.vvpulse)"]
    )
    df["energy_consumed(xtop.vpulse)"] = (
        df["power(xtop.vpulse)"] * df["time"].diff()
    ).cumsum()
    assert "power(xtop.vpulse)" in df.columns
    assert "resistance(xtop.vpulse)" in df.columns
    assert "energy_consumed(xtop.vpulse)" in df.columns
    # Check calculations
    assert df.loc[0, "power(xtop.vpulse)"] == 0.0
    assert df.loc[1, "resistance(xtop.vpulse)"] == 1000
    assert np.isclose(df.loc[10, "energy_consumed(xtop.vpulse)"], 0.0005)


def test_phase_shifter_drive_flow(tmp_path):
    """Test the full flow of driving the phase shifter with simulation data."""
    # Setup phase shifter
    phase_shifter = mzi2x2_2x2_phase_shifter()
    phase_netlist = phase_shifter.get_netlist(exclude_port_types="electrical")
    models = piel.models.frequency.get_default_models()
    mzi2x2_model, mzi2x2_model_info = sax.circuit(netlist=phase_netlist, models=models)

    # Create transient simulation results
    data = {
        "time": np.linspace(0, 0.002, 21),
        "power(xtop.vpulse)": np.linspace(0, 1, 21),
    }
    df = pd.DataFrame(data)

    # Define phase mapping
    phase_map = linear_phase_mapping_relationship(
        phase_power_slope=10, zero_power_phase=1
    )
    df["phase"] = df["power(xtop.vpulse)"].apply(phase_map)

    # Generate unitary matrices
    unitary_array = []
    for phase in df["phase"]:
        unitary = piel.tools.sax.sax_to_s_parameters_standard_matrix(
            mzi2x2_model(sxt={"active_phase_rad": phase}),
            input_ports_order=("o2", "o1"),
        )
        unitary_array.append(unitary)
    df["unitary"] = unitary_array

    # Calculate output amplitudes
    optical_port_input = np.array([1, 0])
    output_amplitude_array_0 = []
    output_amplitude_array_1 = []
    for unitary in df["unitary"]:
        output = np.dot(unitary[0], optical_port_input)
        output_amplitude_array_0.append(output[0])
        output_amplitude_array_1.append(output[1])
    df["output_amplitude_array_0"] = output_amplitude_array_0
    df["output_amplitude_array_1"] = output_amplitude_array_1

    # Verify calculations
    assert "output_amplitude_array_0" in df.columns
    assert "output_amplitude_array_1" in df.columns
    assert len(df["output_amplitude_array_0"]) == len(df)
    assert len(df["output_amplitude_array_1"]) == len(df)


def test_plotting_functions(tmp_path):
    """Test the plotting functions to ensure plots are generated."""
    # Create sample data
    data = {
        "time": np.linspace(0, 1, 100),
        "v(xtop.vpulse_p)": np.sin(np.linspace(0, 2 * np.pi, 100)),
        "i(v.xtop.vvpulse)": np.cos(np.linspace(0, 2 * np.pi, 100)),
    }
    df = pd.DataFrame(data)

    # Test simple plot
    plot = plot_simple(
        x_data=df["time"],
        y_data=df["v(xtop.vpulse_p)"],
        xlabel="Time (s)",
        ylabel="Voltage (V)",
    )
    plot_file = tmp_path / "simple_plot.png"
    plot[0].savefig(plot_file)
    assert plot_file.exists()

    # Test multi-row plot
    multi_plot = plot_simple_multi_row(
        data=df,
        x_axis_column_name="time",
        row_list=["v(xtop.vpulse_p)", "i(v.xtop.vvpulse)"],
        y_label=["Voltage (V)", "Current (A)"],
    )
    multi_plot_file = tmp_path / "multi_row_plot.png"
    multi_plot.savefig(multi_plot_file)
    assert multi_plot_file.exists()


def test_full_cosimulation_flow(tmp_path):
    """Test the complete co-simulation flow from netlist extraction to optical output."""
    # Step 1: Extract electrical netlist
    phase_shifter = mzi2x2_2x2_phase_shifter()
    electrical_netlist = phase_shifter.get_netlist(exclude_port_types="optical")

    # Step 2: Assign resistance
    electrical_netlist["instances"]["sxt"]["info"]["resistance"] = 1000.0

    # Step 3: Create SPICE netlist
    spice_netlist = piel.integration.gdsfactory_netlist_with_hdl21_generators(
        electrical_netlist
    )

    # Step 4: Construct HDL21 module
    circuit = piel.integration.construct_hdl21_module(spice_netlist)

    # Step 5: Configure and run DC simulation
    @h.module
    class OperatingPointTb:
        VSS = h.Port()
        VDC = h.Vdc(dc=1)(n=VSS)
        dut = straight_heater_metal_simple()
        dut.e1 = VDC.p
        dut.e2 = VSS

    sim = configure_operating_point_simulation(
        testbench=OperatingPointTb, name="full_cosim_op_sim"
    )
    op_result = run_simulation(sim)

    # Verify simulation results
    assert "v(xtop.vvdc_p)" in op_result.an[0].data
    assert "i(v.xtop.vvdc)" in op_result.an[0].data
    assert np.isclose(op_result.an[0].data["i(v.xtop.vvdc)"], 0.001, atol=1e-6)

    # Step 6: Configure and run transient simulation
    @h.module
    class TransientTb:
        VSS = h.Port()
        VPULSE = h.Vpulse(
            delay=1 * h.prefix.m,
            v1=-1000 * h.prefix.m,
            v2=1000 * h.prefix.m,
            period=100 * h.prefix.m,
            rise=10 * h.prefix.m,
            fall=10 * h.prefix.m,
            width=75 * h.prefix.m,
        )(n=VSS)
        dut = straight_heater_metal_simple()
        dut.e1 = VPULSE.p
        dut.e2 = VSS

    transient_sim = configure_transient_simulation(
        testbench=TransientTb,
        stop_time_s=0.2,
        step_time_s=1e-4,
        name="full_cosim_transient_sim",
    )
    run_simulation(transient_sim, to_csv=True)

    # Load and verify transient results
    csv_file = tmp_path / "TransientTb.csv"
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert "time" in df.columns
    assert "v(xtop.vpulse_p)" in df.columns
    assert "i(v.xtop.vvpulse)" in df.columns

    # Step 7: Apply phase mapping
    phase_map = linear_phase_mapping_relationship(
        phase_power_slope=10, zero_power_phase=1
    )
    df["phase"] = df["power(xtop.vpulse)"] = (
        df["v(xtop.vpulse_p)"] * df["i(v.xtop.vvpulse)"]
    )
    df["phase"] = df["power(xtop.vpulse)"].apply(phase_map)

    # Step 8: Generate optical unitary matrices
    models = get_default_models()
    phase_netlist = phase_shifter.get_netlist(exclude_port_types="electrical")
    mzi2x2_model, _ = sax.circuit(netlist=phase_netlist, models=models)
    df["unitary"] = df["phase"].apply(
        lambda p: piel.tools.sax.sax_to_s_parameters_standard_matrix(
            mzi2x2_model(sxt={"active_phase_rad": p}),
            input_ports_order=("o2", "o1"),
        )
    )

    # Step 9: Calculate optical outputs
    optical_input = np.array([1, 0])
    df["output_o3"] = df["unitary"].apply(lambda U: np.dot(U[0], optical_input)[0])
    df["output_o4"] = df["unitary"].apply(lambda U: np.dot(U[1], optical_input)[0])

    # Verify optical outputs
    assert "output_o3" in df.columns
    assert "output_o4" in df.columns
    # Simple check: output amplitude should be within expected range
    assert df["output_o3"].abs().max() <= 1.0
    assert df["output_o4"].abs().max() <= 1.0
