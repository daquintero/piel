# # Mixed-Signal & Photonic Cosimulation

# +
import hdl21 as h
import gdsfactory as gf
import piel
import sky130
import sky130_hdl21

from bokeh.io import output_notebook
from gdsfactory.config import rich_output
from gplugins.schematic_editor import SchematicEditor

gf.config.rich_output()

# %env BOKEH_ALLOW_WS_ORIGIN=*

output_notebook()


# -

# ## An Inverter-Driven Thermo-Optic Switch

# In this example, we will implement a basic co-simulation between photonics and electronics.

# Let's consider our thermo-optic resistive load:


def our_custom_resistor_hdl21_module():
    # TODO add custom PDK
    from piel import straight_heater_metal_simple
    import functools
    from gdsfactory.generic_tech import get_generic_pdk

    generic_pdk = get_generic_pdk()
    generic_pdk.activate()

    # Defines the resistance parameters
    our_resistive_heater = functools.partial(
        straight_heater_metal_simple, ohms_per_square=2
    )

    our_resistive_heater_netlist = our_resistive_heater().get_netlist(
        allow_multiple=True, exclude_port_types="optical"
    )

    our_resistive_heater_spice_netlist = piel.gdsfactory_netlist_with_hdl21_generators(
        our_resistive_heater_netlist
    )
    our_resistive_heater_circuit = piel.construct_hdl21_module(
        spice_netlist=our_resistive_heater_spice_netlist
    )
    return our_resistive_heater_circuit


# +
# our_custom_resistor_hdl21_module()
# -

# First, we implement the SPICE model of our inverter in the sky130 process as continuing from a previous example.


@h.module
class SkyInv:
    """An inverter, demonstrating using PDK modules"""

    # Create some IO
    i, o, VDD, VSS = h.Ports(4)

    p = sky130_hdl21.Sky130MosParams(w=1, l=1)

    # And create some transistors!
    ps = sky130_hdl21.primitives.PMOS_1p8V_STD(p)(d=o, g=i, s=VDD, b=VDD)
    ns = sky130_hdl21.primitives.NMOS_1p8V_STD(p)(d=VSS, g=i, s=o, b=VSS)


# Let's configure a basic DC-sweep simulation of this inverter driving a resistive load, which would be equivalent to our thermo-optic switch


@h.module
class OperatingPointTb:
    """# Basic Extracted Device DC Operating Point Testbench"""

    VSS = h.Port()  # The testbench interface: sole port VSS - GROUND
    VDD = h.Vdc(dc=1)(n=VSS)  # A DC voltage source

    load_resistor = sky130_hdl21.ress["GEN_PO"]
    load_resistor.n = VSS

    inv = SkyInv(i=VDD.p, VDD=VDD.p, VSS=VSS)
    load_resistor.p = inv.o


simple_operating_point_simulation = piel.configure_operating_point_simulation(
    testbench=OperatingPointTb, name="simple_operating_point_simulation"
)
results = piel.run_simulation(simulation=simple_operating_point_simulation)
results

# ## A DAC-Driven Mixed Signal Simulation

# #### Automation

# Now, these transient simulations are something you might want to very configure depending on the type of signals that you might want to verify. However, we can provide some basic parameterised simple functions such as step responses and so on. So instead of having to write everything above, you can also just run the following.

# One desired output of an electrical model simulation is an extraction of the power consumption of the circuit. Fundamentally, this is dependent on the time and the operation performed. Hence, to estimate an average power consumption, it is more effective to define the power consumption of a particular operation, and extract the power consumption for the frequency at which this operation is performed.
#
# In this case, we are defining the energy of the operation at particular nodes of the circuit. For example, we know a resisitve heater will dissipate all of its current consumption as thermal power. However, we also need to evaluate the whole circuit. We can know how much energy our DC or RF power supply is providing by measuring the voltage and current supplied accordingly. In a digital circuit, depending on the frequency of the operation, we know how often there is a signal rise and fall, in both cases forcing digital transistors to operate in the linear regime and consuming more power than in saturation mode. We also need to account the range of time the signals are in saturation mode, as even in CMOS idle state there is a minimal power consumption that is important as the circuit scales into VLSI/EP.
#
# Note that through the SPICE simulations, we can extract the energy required at each operation with greater accuracy than analytically and the complexity of this is configuring the testbench appropriately in order to account for this.

#

#

#
