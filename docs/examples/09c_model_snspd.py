# # SNSPD Parametric Modelling
#
# In this example, we will explore some parametric modelling of superconducting-nanowire single-photon detectors in the context of the electronic interconnection requirements and effects.

# ## Create a SNSPD MOdel
#
# Let's start by first composing the SPICE model of a superconducting-nanowire single-photon detector. Let's get the equivalent circuit model:
#
# ![snspd_circuit_model](../_static/img/sections/codesign/devices/snspd.png)
#
# TODO add reference.

# One of the main caveats is how we model the detection event of a trigger photon in the context of a active electronic system. We could, for example, model the detection event as a time-dependent-variable resistor which changes resistance from superconducting to real resistance. In the future, we might want to trigger the electronic model externally from a given input, in which case we'd be inherently modelling something like a mixed-signal circuit.
#
# One way to model this accurately is assume the superconducting resistance $R_{snspd}$ consists of an always-on switch and a real resistance in parallel. Whenever the photon is detected, the switch short is deactivated (say a PMOS turning off) and the superconducting current flows through the real-resistance.

# +
import sys
import hdl21 as h
from hdl21.prefix import µ
from hdl21.primitives import Nmos, Pmos, MosVth


@h.module
class Inv:
    """An inverter, demonstrating instantiating PDK modules"""

    # Create some IO
    i, o, VDD, VSS = h.Ports(4)

    # And now create some generic transistors!
    ps = Pmos(w=1 * µ, l=1 * µ, vth=MosVth.STD)(d=o, g=i, s=VDD, b=VDD)
    ns = Nmos(w=1 * µ, l=1 * µ, vth=MosVth.STD)(d=o, g=i, s=VSS, b=VSS)


h.netlist(Inv(), dest=sys.stdout, fmt="spice")

# +
import hdl21 as h
import hdl21.primitives as hp
import sys


@h.paramclass
class SNSPDParameters:
    """
    These are all the potential parametric configuration variables
    """

    pass


@h.generator
def basic_snspd(params: SNSPDParameters) -> h.Module:
    @h.module
    class BasicSNSPD:
        # Define the ports
        vout = h.Inout()
        vss = h.Inout()
        photon_trigger = h.Input()

        # Define components and parameters
        lk_inductor = hp.Inductor(l=10e-9)(p=vout)  # 10 nH Inductor
        r_term = hp.Resistor(r=50)(p=vout, n=vss)  # 50 Ohm Resistor

        # Instead of a:
        # r_snspd = hp.Resistor(r=5e3)(n=vss) # superconducting hotspot resistor
        # We can have an "equivalent" version:
        r_hot_snspd = hp.Resistor(r=7e3)(n=vss)
        superconducting_switch = hp.Pmos(w=1 * µ, l=1 * µ, vth=MosVth.STD)(
            d=vss, g=photon_trigger, b=vss
        )

        # Connect the switch and resistor in parallel
        superconducting_switch.s = lk_inductor.n
        superconducting_switch.b = lk_inductor.n
        r_hot_snspd.p = lk_inductor.n

    return BasicSNSPD


# -

h.netlist(basic_snspd(), dest=sys.stdout)

# ### Basic Operation Modelling
#
# Let's visualise in the time-domain the operation of a single-photon detector in this context.


# ## Parametric Analysis

# Now, because of the power of python, we are able to do some very interesting circuit-modelling exploration of our devices. Firs
