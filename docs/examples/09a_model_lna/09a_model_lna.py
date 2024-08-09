# # Low-Noise Amplifier Design Principles

# In this example, we will understand the design, modelling and performance requirements of a low-noise amplifier. We will specifically explore ultra-wideband ~180nm CMOS designs as these are more reproducible with open-source toolsets and relevant to some photonics time-frames.
#
# We will first understand performance parameters of a set of ultra-wideband designs before using this as reference to model a similar design using open-source technology and `piel`.

import piel.experimental as pe

# ## Amplifier Characterization
#
# ### Basic DC Amplfier Context
#
# Not all RF LNAs have a bandwidth starting from DC as this is topology dependent (many don't in fact). It can be useful to illustrate some RF amplifier concepts to the radio-frequency terminology uninitated by understanding the relationships to the terminology to used to commonly describe DC amplifiers.
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

pe.visual.plot_dc_sweep(dc_sweep=dc_sweep[0])

pe.visual.plot_dc_sweeps(dc_sweep_collection=dc_sweep)
