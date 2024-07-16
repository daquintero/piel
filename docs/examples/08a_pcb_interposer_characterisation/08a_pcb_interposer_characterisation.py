# # Modelling and Experimental RF Characterization of a PCB Interposer

import piel
import piel.experimental as pe

# In this example, we will compare experimental measurements and simulated measurements to understand the performance of a cryogenic-designed EIC-interposer printed-circuit board.
#
# We will:
#
# - Compare the simulated design characterisitcs with propagation measurements of the device.
# - Understand how to perform cryo-compensation of a microstrip design and compare between cryogenic and room-temperature results.
# - Perform de-embedding and propagation delay measurements of the microstrips to EIC pads.
# - Demonstrate how the `piel` experimental functionality can help implement this.
#
#
# <figure>
# <img src="../../_static/img/examples/08a_pcb_interposer_characterisation/pcb_interposer.jpg" alt="drawing" width="50%"/>
# <figcaption align = "center"> YOUR CAPTION </figcaption>
# </figure>

# ## Creating an `Experiment`
#
# Whenever using an experimental setup, it can be difficult to remember all the configurations that need to be tested with multiple parameters, a set of wiring, and time setups and more. This is especially pressing if the experiment cannot be automated and requires manual input. As such, there is some functionality built into the `piel.experimental` module to help track, manage and compare data with simulated data accordingly.
#
# Let's go through some examples as we talk about the type of characterization we want to perform.
#
# The way this functionality works, is that we create an `Experiment`. This class, like any other pydantic class, can be serialised into a specific individual data serializable configuration which corresponds to `ExperimentInstances`. These are specific data collection points within a larger `Experiment`. This functionality can also be used to create a data set that describes each of these data collection points, and corresponding directories where the data can be stored and managed properly. Let's run through this circumstance.
#




# ## Frequency-Domain Analysis
#
# ## Performing the VNA/Deembedding Calibration
#
# We will use the calibration kit of an Agilent E8364A PNA which is nominally designed for 2.4mm coaxial cables. Using 2.4mm to 3.5mm SMA adapters is probably fine given that we're deembedding up to a given cable. Realistically, I would need to deembed the performance of the adapter between the calibration kit and the open, short, through adapter.
#
# I've saved the calibration under `files/vna_calibration/calibation_35mm_calkit.cst` which can be reloaded. Note it's only useful up to 20GHz. Standard amount of points is 6401 between 0-20 GHz exactly in the frequency spectrum which is about 3MHz resolution per point. Only use GHz.
#
# * TODO list of equipment
#

# * S1-S2 Through
# * S6-S7 Load 50 $\Omega$
#
# ### Through S-Parameter Measurement

# ## Exploring Software Calibration

# Let's assume we have already applied a calibration onto our hardware. In order to implement the `short` and `open`


import skrf as rf
from skrf.calibration import OpenShort

op = rf.Network("open_ckt.s2p")
sh = rf.Network("short_ckt.s2p")
dut = rf.Network("full_ckt.s2p")
dm = OpenShort(dummy_open=op, dummy_short=sh, name="test_openshort")
realdut = dm.deembed(dut)

# ## Time-Domain Analysis
#
# Let's consider we want to measure the propagation velocity of a pulse through one of our coaxial cables. If you are doing a similar experiment, make sure to use ground ESD straps to avoid damage to the equipment. As there is frequency dispersion in the RF transmission lines, we also know the time-domain response is different according to the type of signal applied to the device. We can compare an analysis between the different pulse frequencies.
#
# First, let's consolidate the relevant files in a way we can index and analyse.

pcb_analysis_data = [
    pe.types.PropagationDelayFileCollection(
        device_name="eic_interposer_pcb",
        measurement_name="through_s1_s2",
        reference_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_1GHz_Ch1.csv",
        device_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_1GHz_Ch2.csv",
        measurement_file="measurement_data/through_pcb/mdata_through_ch1ref_ch2pcb_1GHz.csv",
        source_frequency_GHz=1,
    ),
    pe.types.PropagationDelayFileCollection(
        device_name="eic_interposer_pcb",
        measurement_name="through_s1_s2",
        reference_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_3GHz_Ch1.csv",
        device_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_3GHz_Ch2.csv",
        measurement_file="measurement_data/through_pcb/mdata_through_ch1ref_ch2pcb_3GHz.csv",
        source_frequency_GHz=3,
    ),
    pe.types.PropagationDelayFileCollection(
        device_name="eic_interposer_pcb",
        measurement_name="through_s1_s2",
        reference_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_5GHz_Ch1.csv",
        device_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_5GHz_Ch2.csv",
        measurement_file="measurement_data/through_pcb/mdata_through_ch1ref_ch2pcb_5GHz.csv",
        source_frequency_GHz=5,
    ),
    pe.types.PropagationDelayFileCollection(
        device_name="eic_interposer_pcb",
        measurement_name="through_s1_s2",
        reference_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_10GHz_Ch1.csv",
        device_waveform="measurement_data/through_pcb/through_ch1ref_ch2pcb_10GHz_Ch2.csv",
        measurement_file="measurement_data/through_pcb/mdata_through_ch1ref_ch2pcb_10GHz.csv",
        source_frequency_GHz=10,
    ),
]


pcb_propagation_frequency_sweep = pe.types.PropagationDelaySweepFileCollection(
    sweep_parameter_name="source_frequency_GHz",
    files=pcb_analysis_data,
)

# Now we need to write some functionality to extract the files stored in these files in a meaningful way. Fortunately, there's already some functionality using `piel` in this context:

propagation_delay_sweep_data = pe.DPO73304.extract_file_collection_data(
    pcb_propagation_frequency_sweep
)
# propagation_delay_sweep_data

propagation_delay_sweep_data

# Now, we want to plot this files as a function of the sweep parameter. Fortunately this is pretty easy:


fig, ax = piel.visual.plot_signal_propagation_sweep_measurement(
    propagation_delay_sweep_data,
)

fig, ax = piel.visual.plot_signal_propagation_sweep_measurement(
    propagation_delay_sweep_data,
    measurement_name="delay_ch1_ch2__s_2",
)

fig, ax = piel.visual.plot_signal_propagation_sweep_measurement(
    propagation_delay_sweep_data,
    measurement_name="delay_ch1_ch2__s_1",
)

fig, ax = piel.visual.plot_signal_propagation_sweep_measurement(
    propagation_delay_sweep_data,
    measurement_name="pk-pk_ch1__v",
)

# TODO add graph showing this exact setup.
#
# In this setup, we will use a RF signal generator and a RF oscilloscope.
#
# First, we will split the signal generator signal through two paths and see them in the oscillscope. They should overlap each other perfectly. Both signals are terminated at the oscilloscope inputs in order to get an exact rising edge.
