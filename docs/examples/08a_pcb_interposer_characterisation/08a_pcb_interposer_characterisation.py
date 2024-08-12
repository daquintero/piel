# # Modelling and Experimental RF Characterization of a PCB Interposer

import piel
import piel.experimental as pe
from datetime import datetime

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

# ## Creating an `ExperimentInstance`
#
# Whenever using an experimental setup, it can be difficult to remember all the configurations that need to be tested with multiple parameters, a set of wiring, and time setups and more. This is especially pressing if the experiment cannot be automated and requires manual input. As such, there is some functionality built into the `piel.experimental` module to help track, manage and compare data with simulated data accordingly.
#
# Let's go through some examples as we talk about the type of characterization we want to perform.
#
# The way this functionality works, is that we create an `ExperimentInstance`. This class, like any other pydantic class, can be serialised into a specific individual data serializable configuration which corresponds to `ExperimentInstances`. These are specific data collection points within a larger `ExperimentInstance`. This functionality can also be used to create a data set that describes each of these data collection points, and corresponding directories where the data can be stored and managed properly. Let's run through this circumstance.


# First, let's create environmental metadata.

room_temperature_environment = piel.types.Environment(temperature_K=273)


# Now we can create our custom `PCB` definition.
#
# ```
# S6-S7 Load 50 pcb 3
# S1-S2 Through
# RES1 Short to GND
# S8 OPEN
# ```


# +
def pcb_smp_connector(name, pcb_name):
    """
    This is our PCB SMP Port definition factory.
    """
    return piel.types.PhysicalPort(
        name=name, connector="smp_plug", domain="RF", parent_component_name=pcb_name
    )


# These are the measurements we want to check
measurement_connections = {
    "load_through": ("SIG6", "SIG7"),
    "throguh": ("SIG1", "SIG2"),
}


rf_calibration_pcb = piel.models.physical.electrical.create_pcb(
    port_name_list=[
        "SIG14",
        "RES1",
        "SIG1",
        "SIG2",
        "RES2",
        "SIG3",
        "OPEN",
        "SHORT",
        "SIG5",
        "RES3",
        "SIG6",
        "SIG7",
        "RES4",
        "SIG8",
        "L50",
        "GND",
    ],
    connection_tuple_list=[],
    port_factory=pcb_smp_connector,
    pcb_name="PCB3",
    environment=room_temperature_environment,
    components=[],
)

rf_calibration_pcb
# -

# ```python
# PCB(name='PCB3', ports=[PhysicalPort(name='SIG14', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG2', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES2', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG3', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='OPEN', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SHORT', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG5', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES3', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG6', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG7', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES4', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG8', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='L50', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='GND', domain='RF', connector='smp_plug', manifold=None)], connections=[PhysicalConnection(connections=[Connection(name=None, ports=(PhysicalPort(name='SIG6', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG7', domain='RF', connector='smp_plug', manifold=None)))], components=None), PhysicalConnection(connections=[Connection(name=None, ports=(PhysicalPort(name='SIG1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG2', domain='RF', connector='smp_plug', manifold=None)))], components=None), PhysicalConnection(connections=[Connection(name=None, ports=(PhysicalPort(name='RES1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='GND', domain='RF', connector='smp_plug', manifold=None)))], components=None)], components=[], environment=Environment(temperature_K=273.0, region=None), manufacturer=None, model=None)
# ```

# A `Component` might contain subcomponents and there are parameters like `Environment`. In any case, we have all the flexibility of `python` of composing all the `ExperimentInstances` we want. As long as we create an `Experiment` with multiple `ExperimentInstances`, then it is pretty straightforward to just export that to a JSON model file. Using `pydantic`, we can also reinstantiate the model back into `python` which is quite powerful for this type of experiment management.

# ## Creating a Serializable `Experiment`

# ### Frequency-Domain Analysis

# It is possible to extract the time-domain performance from frequency-domain measurements:
#
# - Time Domain Analysis Using a Network Analyzer, Keysight, Application Note
# - [scikit-rf Time Domain and Gating](https://scikit-rf.readthedocs.io/en/latest/examples/networktheory/Time%20Domain.html)
#
# Now, let's define an experiment accordingly. In this experiment, we will do s-parameter measurmeents of two shorted PCB traces directly from a reference plane.


def create_calibration_vna_experiments(measurements: dict, **kwargs):
    """
    Simple two port measurement experiment.
    """
    experiment_instances = list()
    i = 0
    for measurement_i in measurements.items():
        sparameter_measurement_configuration = (
            pe.types.VNASParameterMeasurementConfiguration(
                test_port_power_dBm=-17,
                sweep_points=6401,
                frequency_range_Hz=(45e6, 20e9),
            )
        )

        # Define experimental components
        vna_configuration = pe.types.VNAConfiguration(
            measurement_configuration=sparameter_measurement_configuration
        )

        vna = pe.models.vna.E8364A(configuration=vna_configuration)
        blue_extension_cable = pe.models.cables.generic_sma(
            name="blue_extension", model="1251C", length_m=0.025
        )
        experiment_components = [vna, blue_extension_cable, rf_calibration_pcb]

        # Connect the iteration ports
        measurement_name = measurement_i[0]
        measurement_port1 = measurement_i[1][0]
        measurement_port2 = measurement_i[1][1]

        # Create the VNA connectivity
        experiment_connections = piel.create_component_connections(
            components=experiment_components,
            connection_reference_str_list=[
                [
                    f"{vna.name}.PORT1",
                    f"{blue_extension_cable.name}.IN",
                ],
                [
                    f"{blue_extension_cable.name}.OUT",
                    f"{rf_calibration_pcb.name}.{measurement_port1}",
                ],
                [
                    f"{rf_calibration_pcb.name}.{measurement_port2}",
                    f"{vna.name}.PORT2",
                ],
            ],
        )

        # Define experiment with connections
        experiment_measurement = pe.types.ExperimentInstance(
            name=measurement_name,
            components=experiment_components,
            connections=experiment_connections,
            index=i,
            date_configured=str(datetime.now()),
            measurement_configuration_list=[sparameter_measurement_configuration],
        )

        experiment_instances.append(experiment_measurement)
        i += 1

    return pe.types.Experiment(experiment_instances=experiment_instances, **kwargs)


vna_pcb_experiment = create_calibration_vna_experiments(
    name="pcb_rf_vna_measurement",
    measurements=measurement_connections,
    goal="Perform S-Parameter characterization of a PCB trace.",
)
# vna_pcb_experiment

measurement_connections

# Now, let's create an experiment `data` directory in which to save the data accordingly:

experiment_data_directory = piel.return_path("data")
piel.create_new_directory(experiment_data_directory)

# Now, the beautiful thing that this can do, especially if you are allergic to repettitive experimental tasks like me, is that we can use this to identify all our experimental instances which correspond to specific dials and operating points we need to configure manually.


vna_pcb_experiment_directory = pe.construct_experiment_directories(
    experiment=vna_pcb_experiment,
    parent_directory=experiment_data_directory,
    construct_directory=True,
)

# ```
# Experiment directory created at /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_rf_vna_measurement
# ```

# Let's see how the structure of the project looks like:


# !pwd $vna_pcb_experiment_directory
# !ls $vna_pcb_experiment_directory

# ```bash
# /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation
# 0  1  experiment.json  README.md
# ```

# Now, let's save the experimental data in there accordingly. Once we save the data, we can recompose the data into measurement containers based on the `MeasurementConfigurationTypes` we defined for each `ExperimantInstance`.


example_measurement = pe.compose_measurement_from_experiment_instance(
    vna_pcb_experiment.experiment_instances[1],
    instance_directory=vna_pcb_experiment_directory / "1",
)
example_measurement

# However, we might want to compose our measurements into a `MeasurementCollection`:

vna_pcb_experiment_collection = pe.compose_measurement_collection_from_experiment(
    vna_pcb_experiment,
    experiment_directory=vna_pcb_experiment_directory,
)
vna_pcb_experiment_collection

# ```
# [VNASParameterMeasurement(name='load_through', parent_directory=PosixPath('/home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_rf_vna_measurement/0'), spectrum_file=PosixPath('/home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_rf_vna_measurement/0/through_293K_s1s2_wbluec_50oihm_cjhip.s2p')),
#  VNASParameterMeasurement(name='throguh', parent_directory=PosixPath('/home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_rf_vna_measurement/1'), spectrum_file=PosixPath('/home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_rf_vna_measurement/1/single_trace_293K_s1s2_wbluec.s2p'))]
#
# ```

s_parameter_measurement_data_sweep = pe.extract_data_from_measurement_collection(
    measurement_collection=vna_pcb_experiment_collection,
)
s_parameter_measurement_data_sweep

# We can analyse each of these networks.
#
# We can first understand the `scikit-rf` network configuration:

s_parameter_measurement_data_sweep.collection[0].network

# Let's plot the basic s-parameter `dB` magnitude response:

s_parameter_measurement_data_sweep.collection[0].network.plot_s_db(0, 0)

# Let's plot the two s-parameter measurement data in a basic form:

import matplotlib.pyplot as plt

s11_network = s_parameter_measurement_data_sweep.collection[1].network.subnetwork(
    ports=[0]
)
s21_network = s_parameter_measurement_data_sweep.collection[1].network.subnetwork(
    ports=[1]
)
s11_time, s11_signal = s11_network.step_response()
s21_time, s21_signal = s21_network.step_response()
plt.plot(s11_time, s11_signal)
plt.plot(s21_time, s21_signal)

#  Now this is not particularly useful on its own. It'd be nicer if we can do some more programmatic analysis our our sweep data.

pe.visual.frequency.measurement_data_collection.plot_s_parameter_measurements_to_step_responses(
    data_collection=s_parameter_measurement_data_sweep,
    network_port_index=0,
    time_range_s=(-0.5e-9, 2e-9),
    path=None,
)

# ### Time-Domain Analysis: DUT & Reference Paths
#
# Let's consider we want to measure the propagation velocity of a pulse through one of our coaxial cables. If you are doing a similar experiment, make sure to use ground ESD straps to avoid damage to the equipment. As there is frequency dispersion in the RF transmission lines, we also know the time-domain response is different according to the type of signal applied to the device. We can compare an analysis between the different pulse frequencies.
#
# Let's configure the propagation delay experimental measurement in order to save the files in a reasonable location. We need to define how a specific experiment instance, in this case a measurement looks like. This involves the device configuration and stimulus parameters.


# TODO make a diagram
#
# First, we will do the exact test between two identical set of cables.


def calibration_propagation_delay_experiment_instance(
    square_wave_frequency_Hz: float,
):
    oscilloscope = pe.models.create_two_port_oscilloscope()
    waveform_generator = pe.models.create_one_port_square_wave_waveform_generator(
        peak_to_peak_voltage_V=0.5,
        rise_time_s=1,
        fall_time_s=1,
        frequency_Hz=square_wave_frequency_Hz,
    )
    splitter = pe.models.create_power_splitter_1to2()

    # List of connections
    experiment_connections = piel.create_connection_list_from_ports_lists(
        [
            [splitter.ports[1], oscilloscope.ports[0]],
            [splitter.ports[2], oscilloscope.ports[1]],
        ]
    )

    experiment_instance = pe.types.ExperimentInstance(
        name=f"calibration_{square_wave_frequency_Hz}_Hz",
        components=[oscilloscope, waveform_generator, splitter],
        connections=experiment_connections,
        parameters={"square_wave_frequency_Hz": square_wave_frequency_Hz},
    )
    return experiment_instance


# Now, we will add a path difference between the racing signals, with one path going through the PCB shorted through traces.


def pcb_propagation_delay_experiment_instance(
    square_wave_frequency_Hz: float,
):
    # We create out components
    oscilloscope = pe.models.create_two_port_oscilloscope()
    waveform_generator = pe.models.create_one_port_square_wave_waveform_generator(
        peak_to_peak_voltage_V=0.5,
        rise_time_s=1,
        fall_time_s=1,
        frequency_Hz=square_wave_frequency_Hz,
    )
    splitter = pe.models.create_power_splitter_1to2()

    # List of connections
    experiment_connections = piel.create_connection_list_from_ports_lists(
        [
            [splitter.ports[1], oscilloscope.ports[0]],
            [splitter.ports[2], oscilloscope.ports[1]],
        ]
    )

    # If we want the data that will be generated to have automated analysis, we have to specify what type of experiment instance analysis we want
    propagation_delay_configuration = (
        pe.types.PropagationDelayMeasurementConfiguration()
    )

    # We declare the experimental instance.
    experiment_instance = pe.types.ExperimentInstance(
        name=f"pcb_{square_wave_frequency_Hz}_Hz",
        components=[oscilloscope, waveform_generator, splitter],
        connections=experiment_connections,
        measurement_configuration_list=[propagation_delay_configuration],
    )
    return experiment_instance


oscilloscope = pe.models.create_two_port_oscilloscope()
oscilloscope

# Now let's actually create our `Experiment`.
#
# We want to create an `Experiment` according to our data analysis. It will be easier to understand measurements comparing a `PCB` trace and an identical `calibration` set of cables.


# We will test the propagation response at multiple frequencies. Use a through connection to measure the approximate propagation delay through the calibration cables and PCB trace.


def pcb_propagation_delay_experiment(square_wave_frequency_Hz_list: list[float] = None):
    # Create reference iteration parameters
    parameters_list = list()

    # Create all the experiment instances
    experiment_instance_list = list()

    # Iterate through experimental parameters
    for square_wave_frequency_Hz_i in square_wave_frequency_Hz_list:
        pcb_experiment_instance_i = pcb_propagation_delay_experiment_instance(
            square_wave_frequency_Hz=square_wave_frequency_Hz_i
        )
        experiment_instance_list.append(pcb_experiment_instance_i)
        parameters_list.append({"square_wave_frequency_Hz": square_wave_frequency_Hz_i})

    experiment = pe.types.Experiment(
        name="pcb_multi_frequency_through_propagation_measurement",
        experiment_instances=experiment_instance_list,
        goal="Test the propagation response at multiple frequencies. Use a through connection to measure the approximate propagation delay through the calibration cables and PCB trace.",
        parameters_list=parameters_list,
    )
    return experiment


# We also want to connect the two interconnect cable paths without a DUT to measure how identical they are, as a reference measurement of our device accuracy.


def calibration_propagation_delay_experiment(
    square_wave_frequency_Hz_list: list[float] = None,
):
    # Create reference iteration parameters
    parameters_list = list()

    # Create all the experiment instances
    experiment_instance_list = list()

    # Iterate through experimental parameters
    for square_wave_frequency_Hz_i in square_wave_frequency_Hz_list:
        calibration_experiment_instance_i = (
            calibration_propagation_delay_experiment_instance(
                square_wave_frequency_Hz=square_wave_frequency_Hz_i
            )
        )
        experiment_instance_list.append(calibration_experiment_instance_i)
        parameters_list.append({"square_wave_frequency_Hz": square_wave_frequency_Hz_i})

    experiment = pe.types.Experiment(
        name="calibration_multi_frequency_through_propagation_measurement",
        experiment_instances=experiment_instance_list,
        goal="Test the propagation response at multiple frequencies through interconnect cables. Use a through connection to measure the approximate propagation delay between identical cables.",
        parameters_list=parameters_list,
    )
    return experiment


pcb_propagation_delay_experiment_setup = pcb_propagation_delay_experiment(
    square_wave_frequency_Hz_list=[1e9, 3e9, 5e9, 10e9]
)
calibration_propagation_delay_experiment_setup = (
    calibration_propagation_delay_experiment(
        square_wave_frequency_Hz_list=[1e9, 3e9, 5e9, 10e9]
    )
)

# Now, let's create the experiment directory structure with the corresponding experiment instances

pcb_propagation_delay_experiment_directory = pe.construct_experiment_directories(
    experiment=pcb_propagation_delay_experiment_setup,
    parent_directory=experiment_data_directory,
    construct_directory=True,
)
calibration_propagation_delay_experiment_directory = (
    pe.construct_experiment_directories(
        experiment=calibration_propagation_delay_experiment_setup,
        parent_directory=experiment_data_directory,
        construct_directory=True,
    )
)

# ```
# Experiment directory created at /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/pcb_multi_frequency_through_propagation_measurement
# Experiment directory created at /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/calibration_multi_frequency_through_propagation_measurement
# ```

# ## Performing `ExperimentData` Analysis and Plotting

# We can see in each directory the generated directories and files accordingly. Now we can use this directory to save and consolidate all the metadata of our experiments accordingly.
#
# I've already done it for the experiment as described in this code, so let's explore the data using `piel` accordingly.

# First, let's consolidate the relevant files in a way we can index and analyse. In this case I have done this manually, but of course this can be automated with proper file naming in mind.

# +
pcb_propagation_data = pe.types.PropagationDelayMeasurementCollection(
    collection=[
        pe.types.PropagationDelayMeasurement(
            parent_directory=pcb_propagation_delay_experiment_directory / str(0),
            reference_waveform_file="through_ch1ref_ch2pcb_1GHz_Ch1.csv",
            dut_waveform_file="through_ch1ref_ch2pcb_1GHz_Ch2.csv",
            measurements_file="mdata_through_ch1ref_ch2pcb_1GHz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=pcb_propagation_delay_experiment_directory / str(1),
            reference_waveform_file="through_ch1ref_ch2pcb_3GHz_Ch1.csv",
            dut_waveform_file="through_ch1ref_ch2pcb_3GHz_Ch2.csv",
            measurements_file="mdata_through_ch1ref_ch2pcb_3GHz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=pcb_propagation_delay_experiment_directory / str(2),
            reference_waveform_file="through_ch1ref_ch2pcb_5GHz_Ch1.csv",
            dut_waveform_file="through_ch1ref_ch2pcb_5GHz_Ch2.csv",
            measurements_file="mdata_through_ch1ref_ch2pcb_5GHz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=pcb_propagation_delay_experiment_directory / str(3),
            reference_waveform_file="through_ch1ref_ch2pcb_10GHz_Ch1.csv",
            dut_waveform_file="through_ch1ref_ch2pcb_10GHz_Ch2.csv",
            measurements_file="mdata_through_ch1ref_ch2pcb_10GHz.csv",
        ),
    ]
)

calibration_propagation_data = pe.types.PropagationDelayMeasurementCollection(
    collection=[
        pe.types.PropagationDelayMeasurement(
            parent_directory=calibration_propagation_delay_experiment_directory
            / str(0),
            reference_waveform_file="calibration_loop_1Ghz_Ch1.csv",
            dut_waveform_file="calibration_loop_1Ghz_Ch2.csv",
            measurements_file="mdata_calibration_loop_1Ghz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=calibration_propagation_delay_experiment_directory
            / str(1),
            reference_waveform_file="calibration_loop_3Ghz_Ch1.csv",
            dut_waveform_file="calibration_loop_3Ghz_Ch2.csv",
            measurements_file="mdata_calibration_loop_3Ghz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=calibration_propagation_delay_experiment_directory
            / str(2),
            reference_waveform_file="calibration_loop_5Ghz_Ch1.csv",
            dut_waveform_file="calibration_loop_5Ghz_Ch2.csv",
            measurements_file="mdata_calibration_loop_5Ghz.csv",
        ),
        pe.types.PropagationDelayMeasurement(
            parent_directory=calibration_propagation_delay_experiment_directory
            / str(3),
            reference_waveform_file="calibration_loop_10Ghz_Ch1.csv",
            dut_waveform_file="calibration_loop_10Ghz_Ch2.csv",
            measurements_file="mdata_calibration_loop_10Ghz.csv",
        ),
    ]
)
# -


# So these measurements are just the measurement definition, but do not contain the data. We need to extract it from the files.

calibration_propagation_delay_data = (
    pe.DPO73304.extract_propagation_delay_measurement_sweep_data(
        calibration_propagation_data
    )
)
pcb_propagation_delay_data = (
    pe.DPO73304.extract_propagation_delay_measurement_sweep_data(pcb_propagation_data)
)

# Now we need to write some functionality to extract the files stored in these files in a meaningful way. Fortunately, there's already some functionality using `piel` in this context. We will now create a set of `ExperimentData` that represent both the metadata, configuration and data extracted accordingly.

pcb_propagation_delay_experiment_data = pe.types.ExperimentData(
    experiment=pcb_propagation_delay_experiment_setup, data=pcb_propagation_delay_data
)
calibration_propagation_delay_experiment_data = pe.types.ExperimentData(
    experiment=calibration_propagation_delay_experiment_setup,
    data=calibration_propagation_delay_data,
)

# #### Specific Data Plotting Functionality

# Now, we want to plot this files as a function of the sweep parameters. Fortunately this is pretty easy. Let's first start by plotting the signals in time.
#
# TODO add graph showing this exact setup.
#
#
# In this setup, we will use a RF signal generator and a RF oscilloscope.
#
# First, we will split the signal generator signal through two paths and see them in the oscillscope. They should overlap each other perfectly. Both signals are terminated at the oscilloscope inputs in order to get an exact rising edge.


fig, ax = pe.visual.propagation.experiment_data.plot_propagation_signals_time(
    calibration_propagation_delay_experiment_data,
    path="../../_static/img/examples/08a_pcb_interposer_characterisation/calibration_propagation_delay_signals.jpg",
)

# ![calibration_propagation_delay_signals](../../_static/img/examples/08a_pcb_interposer_characterisation/calibration_propagation_delay_signals.jpg)

fig, ax = pe.visual.propagation.experiment_data.plot_propagation_signals_time(
    pcb_propagation_delay_experiment_data,
    path="../../_static/img/examples/08a_pcb_interposer_characterisation/pcb_propagation_delay_signals.jpg",
)


# ![pcb_propagation_delay_signals](../../_static/img/examples/08a_pcb_interposer_characterisation/pcb_propagation_delay_signals.jpg)

# We can also plot the data related to the metrics extracted from the measurements.

fig, ax = pe.visual.propagation.experiment_data.plot_signal_propagation_measurements(
    calibration_propagation_delay_experiment_data,
    x_parameter="square_wave_frequency_Hz",
    measurement_name="delay_ch1_ch2__s_2",
    path="../../_static/img/examples/08a_pcb_interposer_characterisation/calibration_propagation_delay_measurements.jpg",
)

# ![calibration_propagation_delay_measurements](../../_static/img/examples/08a_pcb_interposer_characterisation/calibration_propagation_delay_measurements.jpg)

fig, ax = pe.visual.propagation.experiment_data.plot_signal_propagation_measurements(
    pcb_propagation_delay_experiment_data,
    x_parameter="square_wave_frequency_Hz",
    measurement_name="delay_ch1_ch2__s_1",
    path="../../_static/img/examples/08a_pcb_interposer_characterisation/pcb_propagation_delay_measurements.jpg",
)

# ![pcb_propagation_delay_measurements](../../_static/img/examples/08a_pcb_interposer_characterisation/pcb_propagation_delay_measurements.jpg)

# #### Automatic Report and Plotting

# One of the nice functionalities provided by `piel.experimental` is that because `Experiment`s and `ExperimentData`s can be serialized. That means their analysis can also be automated at multiple stages of the development flow.

# In this example, we have relied on using previous metadata generated in the same python session. We don't always have to do this. Using the generated `experiment.json` or `instance.json` we can straightforwardly reinstantiate our `Experiment` or `ExperimentInstance` models into another instance.

calibration_propagation_delay_experiment_directory_json = (
    calibration_propagation_delay_experiment_directory / "experiment.json"
)
assert calibration_propagation_delay_experiment_directory_json.exists()

# +
# calibration_propagation_delay_experiment_setup
# -

reinsantiated_calibration_experiment = piel.models.load_from_json(
    calibration_propagation_delay_experiment_directory_json, pe.types.Experiment
)
reinsantiated_calibration_experiment

# assert reinsantiated_calibration_experiment == calibration_propagation_delay_experiment_setup
reinsantiated_calibration_experiment.name

# ```bash
# 'calibration_multi_frequency_through_propagation_measurement'
# ```

# Note that this has some limitations of revalidation and reinstantion of python classes.
