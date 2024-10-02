# # Optical Delay Timing Models & Analysis

# A very important aspect of designing concurrent photonic-electronic systems is concurrent operations of optical and electrical signals in time. This means, when we design photonic-electronic systems, we need to account for a higher dimensionality of time matching.
#
# In RF systems, or at least in the design of RF PCBs, a very common concept is path-length matching of transmission lines. The goal is to implement timing equivalence between pulse propagation. In electronic-photonic systems, the complexity is a higher dimensionsionality of various dielectric propagation materials. As such, group velocities of an electromagnetic pulse $v_{g}$ travels at different speeds. Path-length matching on its own is not suitable because of the multiple group velocities involved in interacting pulses. This is particularly a problem for systems that are very sensitive to frequency-dispersion. As such, we need more advanced tools for time-matching.
#
# `piel` has some functionality to achieve this as exemplified in this notebook.

import piel
import piel.experimental as pe

# ## Characterizing a Pulsed Laser ScalarSource

# To measure delay in time, you will probably need a time optical source. In this case, let's create a `PulsedLaser`. We will create a model of this laser:
#
# ![advalue_mll_metrics](../../_static/img/examples/09b_optical_delay/advalue_mll_metrics.png)

mll_2um_metrics = piel.types.PulsedLaserMetrics(
    pulse_power_W=piel.models.metrics.min_max(min=0, max=10e3, unit=piel.types.units.W),
    average_power_W=piel.models.metrics.value(value=1, unit=piel.types.units.W),
    pulse_repetition_rate_Hz=piel.models.metrics.value(
        value=40e6, unit=piel.types.units.Hz
    ),
    pulse_width_s=piel.models.metrics.value(min=3e-12, unit=piel.types.units.s),
)
mll_2um = piel.types.PulsedLaser(name="mll_2um", metrics=[mll_2um_metrics])

# `piel` actually provides some functionality to automatically generate an equivalent signal model of this laser in multiple domains:

laser_pulses_time_data_W = (
    piel.models.transient.electro_optic.generate_laser_time_data_pulses(
        pulsed_laser=mll_2um, time_frame_s=50e-9, point_amount=100000
    )
)

# We can quite easily plot this:

piel.visual.plot.signals.time.plot_time_signal_data(
    laser_pulses_time_data_W,
    path="../../_static/img/examples/09b_optical_delay/example_laser_pulse_time_data.png",
)

# ![example_laser_pulse_time_data](../../_static/img/examples/09b_optical_delay/example_laser_pulse_time_data.png)

# Note you can also use any of the time-domain analysis functionality in piel with this data as you like.

# ## Equivalent Experimental Measurement Analysis

optical_delay_signal = pe.DPO73304.extract_to_data_time_signal(
    "data/example_delay_measurement.csv"
)

piel.visual.plot.signals.time.plot_time_signal_data(
    optical_delay_signal,
    xlabel=piel.types.units.ns,
    title="Raw Measured Optical Pulse Data",
    path="../../_static/img/examples/09b_optical_delay/raw_measured_optical_pulse.png",
)

# ![raw_measured_optical_pulse](../../_static/img/examples/09b_optical_delay/raw_measured_optical_pulse.png)

# ### Extracting the pulse signals above a threshold

pulses_above_threshold_list = piel.analysis.signals.time.extract_signal_above_threshold(
    optical_delay_signal, threshold=0.005
)

piel.visual.plot.signals.time.plot_multi_data_time_signal_different(
    pulses_above_threshold_list,
    xlabel=piel.types.units.ns,
    title="Extracted Pulses above a 0.005 V Threshold",
)

# Note it's interesting these pulses are about 80ps long approximately.

# You can also extract the full pulses:

full_pulses_list = piel.analysis.signals.time.extract_pulses_from_signal(
    optical_delay_signal,
    pre_pulse_time_s=1e-9,
    post_pulse_time_s=1e-9,
)

piel.visual.plot.signals.time.plot_multi_data_time_signal_different(
    full_pulses_list,
    xlabel=piel.types.units.ns,
    title="Extracted Raw Pulses",
    path="../../_static/img/examples/09b_optical_delay/extracted_raw_pulses.png",
)

# ![extracted_raw_pulses](../../_static/img/examples/09b_optical_delay/extracted_raw_pulses.png)

# ### Extracting the noise floor

noise_generator = (
    piel.analysis.signals.time.extract_off_state_generator_from_full_state_data(
        full_time_signal_data=optical_delay_signal,
    )
)
noise_signal = noise_generator(duration_s=1e-9)

piel.visual.plot.signals.time.plot_time_signal_data(
    noise_signal,
    xlabel=piel.types.units.ns,
    title="Example Noise Extraction",
    path="../../_static/img/examples/09b_optical_delay/noise_signal.png",
)

# ![noise_signal](../../_static/img/examples/09b_optical_delay/noise_signal.png)

# ### Separating two signals

high_threshold_pulse_list, low_threshold_pulse_list = (
    piel.analysis.signals.time.separate_per_pulse_threshold(
        optical_delay_signal,
        first_signal_threshold=0.03,
        second_signal_threshold=0.01,
        trigger_delay_s=1e-9,
    )
)

len(high_threshold_pulse_list)

piel.visual.plot.signals.time.plot_multi_data_time_signal_different(
    low_threshold_pulse_list, xlabel=piel.types.ns
)

piel.visual.plot.signals.time.plot_multi_data_time_signal_different(
    low_threshold_pulse_list, xlabel=piel.types.ns
)

# We can easily perform some analysis on these signals:

high_threshold_pkpk_metrics = (
    piel.analysis.signals.time.extract_peak_to_peak_metrics_list(
        high_threshold_pulse_list, unit=piel.types.units.V
    )
)

high_threshold_pkpk_metrics[0:1].table

piel.analysis.metrics.aggregate_scalar_metrics_collection(
    high_threshold_pkpk_metrics
).table

# Now we can recompose this back using our noise generation function.

low_threshold_pulse_signal = piel.analysis.signals.time.compose_pulses_into_signal(
    low_threshold_pulse_list, start_time_s=-50e-9, end_time_s=40e-9
)
high_threshold_pulse_signal = piel.analysis.signals.time.compose_pulses_into_signal(
    high_threshold_pulse_list, start_time_s=-50e-9, end_time_s=40e-9
)

piel.visual.plot.signals.time.plot_multi_data_time_signal_different(
    [low_threshold_pulse_signal, high_threshold_pulse_signal],
    xlabel=piel.types.ns,
    title="Example High-Low Pulse Threshold Signal Decomposition",
    path="../../_static/img/examples/09b_optical_delay/high_low_signal_decomposition.png",
)

# You can also do this analysis all in one:

low_threshold_pulse_signal, high_threshold_pulse_signal = (
    piel.analysis.signals.time.split_compose_per_pulse_threshold(
        signal_data=optical_delay_signal,
        first_signal_threshold=0.03,
        second_signal_threshold=0.01,
        trigger_delay_s=5e-9,
        start_time_s=-50e-9,
        end_time_s=40e-9,
    )
)
