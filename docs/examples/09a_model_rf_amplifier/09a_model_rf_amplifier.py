# # RF Amplifier Design & Analysis Flow

# In this example, we will understand the design, modelling and performance requirements of an RF amplifier. We will specifically explore ultra-wideband ~180nm CMOS designs as these are more reproducible with open-source toolsets and relevant to some photonics-compatible loads.
#
# We will first understand performance parameters of a set of ultra-wideband designs before using this as reference to model a similar design using open-source technology and `piel`.

import piel
import piel.experimental as pe
import os

# ## Basic Terminology
#
# Let's start by understanding basic terminology of our amplification range.
#
# Let's first consider the relationship between DC and RF time-and-frequency units in terms of representing the power. Say we are considering signal sources such as a [SynthHD](https://windfreaktech.com/product/microwave-signal-generator-synthhd/).

# ### Power Conversion Functionality

# Let's consider some power conversion tools which are very common when talking about amplifiers:

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

# ## Automated Performance Metrics Analysis

# `piel` provides multiple analysis functionality in order to compare the metrics of different types of published designs. In this case, we will look into designs in the range of the current open-source EIC technologies. In this case, we will construct and visualise multiple design metrics together. Let's do a little literature review for reference which has already been saved in a parse-able-json as below.

# ### ~180nm Low-Noise Amplifier Metrics Analysis

# Let's assume we have a dataset. You can see how to automate generating one in `regenerate_reference_dataframe.py`. Note you can use any json-generation functionality with a compatible schema.

lna_180nm_metrics = piel.read_json("data/lna_180nm_metrics.json")

# We want to load this into a collection object so that we can do more straightforward analysis:

lna_180nm_collection = piel.types.RFAmplifierCollection.parse_obj(lna_180nm_metrics)
# lna_180nm_collection

# We might want to do some analytics with this amplifier collection:

lna_180nm_performance_table = (
    piel.analysis.electronic.compose_amplifier_collection_performance_dataframe(
        amplifier_collection=lna_180nm_collection,
        desired_metrics=["bandwidth_Hz", "power_gain_dB", "power_consumption_mW"],
    )
)
lna_180nm_performance_table

# |    | name   |   bandwidth_Hz_min |   bandwidth_Hz_max |   power_gain_dB_min |   power_gain_dB_max |   power_consumption_mW_min |   power_consumption_mW_max |
# |---:|:-------|-------------------:|-------------------:|--------------------:|--------------------:|---------------------------:|---------------------------:|
# |  0 |        |            4e+08   |            1e+10   |               11.2  |               12.4  |                      12    |                      12    |
# |  1 |        |            5e+08   |            1.4e+10 |               10.6  |               10.6  |                      52    |                      52    |
# |  2 |        |            1e+08   |            7e+09   |               12.6  |               12.6  |                       0.75 |                       0.75 |
# |  3 |        |            5.7e+09 |            5.7e+09 |               11.45 |               11.45 |                       4    |                       4    |

# We can also quite easily generate a metrics table which could be used in a larger `TeX` document:

lna_180nm_performance_table_tex = piel.visual.table.electronic.compose_amplifier_collection_performance_latex_table(
    amplifier_collection=lna_180nm_collection,
    desired_metrics=["bandwidth_Hz", "power_gain_dB", "power_consumption_mW"],
    caption="Compiled electronic performance available from the best CMOS LNA and PA literature for successful low-noise and power amplification.",
    label="table:amplifier_designs_review",
)
piel.write_file(
    directory_path=os.getenv("TAT"),
    file_text=lna_180nm_performance_table_tex,
    file_name="lna_180nm_metrics_analysis.tex",
)
print(lna_180nm_performance_table_tex)

# ```tex
# \begin{center}
#   \begin{table}[h!]
#       \centering
#       \makebox[\textwidth]{%
#           \begin{tabularx}{0.9\paperwidth}{
# |>{\raggedright\arraybackslash\hsize=\hsize}X|X|X|X|
#           }
#           \hline
#            & \textbf{Bandwidth} (GHz) & \textbf{Power Gain} (dB) & \textbf{Power} (mW) \\
# \hline
# \cite{chen2007ultra} & 0.40 - 10.00 & 11.20 - 12.40 & 12.00 \\
# \hline
# \cite{liu20030} & 0.50 - 14.00 & 10.60 & 52.00 \\
# \hline
# \cite{parvizi2014sub} & 0.10 - 7.00 & 12.60 & 0.75 \\
# \hline
# \cite{asgaran20064} & 5.70 & 11.45 & 4.00 \\
# \hline
#       \end{tabularx}%
#       }
#       \caption{Compiled electronic performance available from the best CMOS LNA and PA literature for successful low-noise and power amplification.}
#       \label{table:amplifier_designs_review}
#   \end{table}
# \end{center}
#
#
# ```

# The power of this API is that now we can programmatically perform analysis on literature designs, and use their performance metrics in a larger system we might want to design. This also means we can compare our own designs in relation to the literature much more straightforwardly. The further benefit of this is that if you are doing a literature review, you can input the analysis data into one data container, and use it for both programmatic analysis and making comparison tables.

# For example, we can now perform visual plots of the parametric performance accordingly in order to provide a level of comparison:

# ### ~180nm Power Amplifier Metrics Analysis

# Let's also look into published power amplifier metrics accordingly:
