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
        
        # Define components and parameters
        lk_inductor = hp.Inductor(l=10e-9)(p=vout)  # 10 nH Inductor
        r_term = hp.Resistor(r=50)(p=vout, n=vss) # 50 Ohm Resistor
        r_snspd = hp.Resistor(r=500)(n=vss) # superconducting hotspot resistor

        lk_inductor.n = r_snspd.p

    return BasicSNSPD


# -

h.netlist(basic_snspd(), dest=sys.stdout)

# ### Basic Operation Modelling
#
# Let's visualise in the time-domain the operation of a single-photon detector in this context.



# ## Parametric Analysis

# Now, because of the power of python, we are able to do some very interesting circuit-modelling exploration of our devices. Firs


