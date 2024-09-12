# # RF Amplifier Design Principles

# In this example, we will understand the design, modelling and performance requirements of an RF amplifier. We will specifically explore ultra-wideband ~180nm CMOS designs as these are more reproducible with open-source toolsets and relevant to some photonics time-frames.
#
# We will first understand performance parameters of a set of ultra-wideband designs before using this as reference to model a similar design using open-source technology and `piel`.

import piel
import piel.experimental as pe

# ## Basic Terminology
#
# Let's start by understanding basic terminology of our amplification range.
#
# Let's first consider the relationship between DC and RF time-and-frequency units in terms of representing the power. Say we are considering signal sources such as a [SynthHD](https://windfreaktech.com/product/microwave-signal-generator-synthhd/).

# ### Power Conversion Functionality

# Let's consider some power conversion tools

# If we want to convert from $dBm$ to $W$:
#
# \begin{equation}
# P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)
# \end{equation}
#
# and backwards:
#
# \begin{equation}
# P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}
# \end{equation}
#

# We can use this functionality from `piel`:

piel.units.dBm2watt(30)

piel.units.watt2dBm(1)

# Now, normally in RF systems we consider things in terms of network impedance as this is in relation to our waveguides and matching loads. We might want to know what is the peak-to-peak voltage of each of our loads accordingly.

# You might want to work out what the peak-to-peak voltage is for a 50 $\Omega$ load for a given decibel-milliwatt measurement:

piel.units.dBm2vpp(1, impedance=50)

piel.units.dBm2vpp(10)

# Or convert backwards from a given $dBm$:

piel.units.vpp2dBm(2)

# ### Relationship to Signal Metrics

# We will explore the performance of a SynthHD (v2): 10MHz – 15GHz Dual Channel Microwave RF Signal Generator in this context.
#
# > Tune any frequency between 10MHz and 15GHz in 0.1Hz resolution. Adjust calibrated amplitude in 0.1dB resolution up to +20dBm and across more than 50dB of range
#
# So the maximum this can output is:

piel.units.dBm2vpp(dBm=20)

# ```
# 6.324555320336759
# ```

# If it has 50dB of range then the minimum power output may be around -30dBm:

piel.units.dBm2vpp(dBm=-30)

# ```
# 0.02
# ```

# ## Design Principles
#
# ### From a Basic DC Amplfier Context
#
# A small subset RF amplifiers have a bandwidth starting from DC as this is topology dependent, and many are narrowband or at a specific bandwidth. It can be useful to illustrate some RF amplifier concepts to the radio-frequency terminology uninitated by understanding the relationships to the terminology to used to commonly describe DC amplifiers.
#
# For example, DC amplifiers such as non-inverting operational amplifier circuits, will have a DC transfer function which maps the gain from inputs to outputs. This gain is frequency dependent and becomes more important in RF regimes, and is characterized differently accordingly using a VNA for example. For now, let's demonstrate some basic DC sweep analysis of a DC amplifier using some `piel` utilities.


# +
# import pandas as pd

# data = pd.read_csv("data/example_dc_response.csv")

# data[data["driver_b_v_set"] == 1.6]
# data["driver_b_v_set"].unique()
# opp = data[["driver_b_v_set"]].drop_duplicates()
# for index, operating_point in opp.iterrows():
#     print(index)
#     print(operating_point)
#     a = data[(data[["driver_b_v_set"]] == operating_point).all(axis=1)]
# print(a)
# -

dc_sweep = pe.extract_dc_sweeps_from_operating_point_csv(
    file_path="data/example_dc_response.csv",
    sourcemeter_voltage_current_signal_name_pairs=[
        ("driver_a_v", "driver_a_i"),
        ("driver_b_v", "driver_b_i"),
    ],
    multimeter_signals=["measurement_a_v"],
    unique_operating_point_columns=["driver_b_v_set"],
)

# piel.visual.plot_dc_sweep(dc_sweep=dc_sweep[0])
#
# piel.visual.plot_dc_sweeps(dc_sweep_collection=dc_sweep)
