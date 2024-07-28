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


# +
def pcb_smp_connector(name):
    """
    This is our PCB SMP Port definition factory.
    """
    return piel.types.PhysicalPort(name=name, connector="smp_plug", domain="RF")


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
    ],
    port_factory=pcb_smp_connector,
    name="rf_calibration_pcb",
    environment=room_temperature_environment,
)

rf_calibration_pcb
# -

# ```python
# PCB(name='cryo_rf_eic_interposer_v1', ports=[PhysicalPort(name='SIG14', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG1', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG2', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES2', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG3', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='OPEN', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SHORT', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG5', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES3', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG6', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG7', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='RES4', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='SIG8', domain='RF', connector='smp_plug', manifold=None), PhysicalPort(name='L50', domain='RF', connector='smp_plug', manifold=None)], connections=[], environment=Environment(temperature_K=273.0, region=None))
# ```

# A `Component` might contain subcomponents and there are parameters like `Environment`. In any case, we have all the flexibility of `python` of composing all the `ExperimentInstances` we want. As long as we create an `Experiment` with multiple `ExperimentInstances`, then it is pretty straightforward to just export that to a JSON model file. Using `pydantic`, we can also reinstantiate the model back into `python` which is quite powerful for this type of experiment management.

# For example, we might do multiple measurements with different relevant for example.


# Now, let's define an experiment accordingly:


def create_vna_measurements(
    vna_configuration: list[pe.types.VNAConfiguration] = None, **kwargs
):
    """
    This is a function that parametrizes multiple VNA measurement configurations and creates a set of VNA measurements
     accordingly.
    """
    if vna_configuration is None:
        vna_configuration = [None]

    experiment_instances = list()
    i = 0
    for vna_configuration_i in vna_configuration:
        # Simple two port measurement
        vna = pe.types.VNA(
            environment=room_temperature_environment,
            name="DPO73304",
            configuration=vna_configuration_i,
        )

        # List of connections
        experiment_connections = []

        # Let's assume that we want to measure an open return loss between SIG14 and RES1
        experiment_connections.extend(
            piel.models.create_all_connections(
                [vna.ports[0], rf_calibration_pcb.ports[0]],
            )
        )

        experiment_connections.extend(
            piel.models.create_all_connections(
                [vna.ports[1], rf_calibration_pcb.ports[1]]
            )
        )

        # Define experiment with connections
        experiment = pe.types.ExperimentInstance(
            components=[rf_calibration_pcb, vna],
            connections=experiment_connections,
            index=i,
            date_configured=datetime.now(),
        )

        experiment_instances.append(experiment)
        i += 1

    return pe.types.Experiment(experiment_instances=experiment_instances, **kwargs)


vna_pcb_experiment = create_vna_measurements(name="basic_vna_test")
vna_pcb_experiment

# Now, let's create an experiment `data` directory in which to save the data accordingly:

experiment_data_directory = piel.return_path("data")
piel.create_new_directory(experiment_data_directory)

# Now, the beautiful thing that this can do, especially if you are allergic to repettitive experimental tasks like me, is that we can use this to identify all our experimental instances which correspond to specific dials and operating points we need to configure manually.


pe.construct_experiment_directories(
    experiment=vna_pcb_experiment, parent_directory=experiment_data_directory
)

# ```
# Experiment directory created at /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/basic_vna_test
# PosixPath('/home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/basic_vna_test')
# ```

# Let's see how the structure of the project looks like:


# ## Time-Domain Analysis
#
# Let's consider we want to measure the propagation velocity of a pulse through one of our coaxial cables. If you are doing a similar experiment, make sure to use ground ESD straps to avoid damage to the equipment. As there is frequency dispersion in the RF transmission lines, we also know the time-domain response is different according to the type of signal applied to the device. We can compare an analysis between the different pulse frequencies.
#
# Let's configure the propagation delay experimental measurement in order to save the files in a reasonable location. We need to define how a specific experiment instance, in this case a measurement looks like. This involves the device configuration and stimulus parameters.


# First, we will do the exact test between two identical set of cables.


def calibration_propagation_delay_experiment_instance(
    square_wave_frequency_Hz: float,
):
    oscilloscope = pe.create_two_port_oscilloscope()
    waveform_generator = pe.create_one_port_square_wave_waveform_generator(
        peak_to_peak_voltage_V=0.5,
        rise_time_s=1,
        fall_time_s=1,
        frequency_Hz=square_wave_frequency_Hz,
    )
    splitter = pe.create_power_splitter_1to2()

    # List of connections
    experiment_connections = piel.models.create_connection_list_from_ports_lists(
        [
            [splitter.ports[1], oscilloscope.ports[0]],
            [splitter.ports[2], oscilloscope.ports[1]],
        ]
    )

    experiment_instance = pe.types.ExperimentInstance(
        name=f"calibration_{square_wave_frequency_Hz}_Hz",
        components=[oscilloscope, waveform_generator, splitter],
        connections=experiment_connections,
    )
    return experiment_instance


# Now, we will add a path difference between the racing signals, with one path going through the PCB shorted through traces.


def pcb_propagation_delay_experiment_instance(
    square_wave_frequency_Hz: float,
):
    oscilloscope = pe.create_two_port_oscilloscope()
    waveform_generator = pe.create_one_port_square_wave_waveform_generator(
        peak_to_peak_voltage_V=0.5,
        rise_time_s=1,
        fall_time_s=1,
        frequency_Hz=square_wave_frequency_Hz,
    )
    splitter = pe.create_power_splitter_1to2()

    # List of connections
    experiment_connections = piel.models.create_connection_list_from_ports_lists(
        [
            [splitter.ports[1], oscilloscope.ports[0]],
            [splitter.ports[2], oscilloscope.ports[1]],
        ]
    )

    experiment_instance = pe.types.ExperimentInstance(
        name=f"pcb_{square_wave_frequency_Hz}_Hz",
        components=[oscilloscope, waveform_generator, splitter],
        connections=experiment_connections,
    )
    return experiment_instance


oscilloscope = pe.create_two_port_oscilloscope()
oscilloscope

# Now let's actually create our `Experiment`:


def propagation_delay_experiment(square_wave_frequency_Hz_list: list[float] = None):
    experiment_instance_list = list()
    for square_wave_frequency_Hz_i in square_wave_frequency_Hz_list:
        pcb_experiment_instance_i = pcb_propagation_delay_experiment_instance(
            square_wave_frequency_Hz=square_wave_frequency_Hz_i
        )
        experiment_instance_list.append(pcb_experiment_instance_i)

        calibration_experiment_instance_i = (
            calibration_propagation_delay_experiment_instance(
                square_wave_frequency_Hz=square_wave_frequency_Hz_i
            )
        )
        experiment_instance_list.append(calibration_experiment_instance_i)

    experiment = pe.types.Experiment(
        name="multi_frequency_through_propagation_measurement",
        experiment_instances=experiment_instance_list,
        goal="Test the propagation response at multiple frequencies. Use a through connection to measure the approximate propagation delay through the calibration cables and PCB trace.",
    )
    return experiment


basic_propagation_delay_experiment = propagation_delay_experiment(
    square_wave_frequency_Hz_list=[1e9, 3e9, 5e9, 10e9]
)

# Now, let's create the experiment directory structure with the corresponding experiment instances

propagation_delay_experiment_directory = pe.construct_experiment_directories(
    experiment=basic_propagation_delay_experiment,
    parent_directory=experiment_data_directory,
)

# ```
# Experiment directory created at /home/daquintero/phd/piel/docs/examples/08a_pcb_interposer_characterisation/data/multi_frequency_through_propagation_measurement
# ```

# We can see in each directory the generated directories and files accordingly. Now we can use this directory to save and consolidate all the metadata of our experiments accordingly.
#
# I've already done it for the experiment as described in this code, so let's explore the data using `piel` accordingly.

# First, let's consolidate the relevant files in a way we can index and analyse. In this case I have done this manually, but of course this can be automated with proper file naming in mind.

# +
pcb_propagation_data = [
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(0),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[0],
        reference_waveform_prefix="through_ch1ref_ch2pcb_1GHz_Ch1.csv",
        dut_waveform_prefix="through_ch1ref_ch2pcb_1GHz_Ch2.csv",
        measurement_file_prefix="mdata_through_ch1ref_ch2pcb_1GHz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(1),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[1],
        reference_waveform_prefix="through_ch1ref_ch2pcb_3GHz_Ch1.csv",
        dut_waveform_prefix="through_ch1ref_ch2pcb_3GHz_Ch2.csv",
        measurement_file_prefix="mdata_through_ch1ref_ch2pcb_3GHz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(2),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[2],
        reference_waveform_prefix="through_ch1ref_ch2pcb_5GHz_Ch1.csv",
        dut_waveform_prefix="through_ch1ref_ch2pcb_5GHz_Ch2.csv",
        measurement_file_prefix="mdata_through_ch1ref_ch2pcb_5GHz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(3),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[3],
        reference_waveform_prefix="through_ch1ref_ch2pcb_10GHz_Ch1.csv",
        dut_waveform_prefix="through_ch1ref_ch2pcb_10GHz_Ch2.csv",
        measurement_file_prefix="mdata_through_ch1ref_ch2pcb_10GHz.csv",
    ),
]

calibration_propagation_data = [
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(4),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[4],
        reference_waveform_prefix="calibration_loop_1Ghz_Ch1.csv",
        dut_waveform_prefix="calibration_loop_1Ghz_Ch2.csv",
        measurement_file_prefix="mdata_calibration_loop_1Ghz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(5),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[5],
        reference_waveform_prefix="calibration_loop_3Ghz_Ch1.csv",
        dut_waveform_prefix="calibration_loop_3Ghz_Ch2.csv",
        measurement_file_prefix="mdata_calibration_loop_3Ghz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(6),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[6],
        reference_waveform_prefix="calibration_loop_5Ghz_Ch1.csv",
        dut_waveform_prefix="calibration_loop_5Ghz_Ch2.csv",
        measurement_file_prefix="mdata_calibration_loop_5Ghz.csv",
    ),
    pe.types.PropagationDelayMeasurement(
        parent_directory=propagation_delay_experiment_directory / str(7),
        experiment_instance=basic_propagation_delay_experiment.experiment_instances[6],
        reference_waveform_prefix="calibration_loop_10Ghz_Ch1.csv",
        dut_waveform_prefix="calibration_loop_10Ghz_Ch2.csv",
        measurement_file_prefix="mdata_calibration_loop_10Ghz.csv",
    ),
]
# -


# Now we need to write some functionality to extract the files stored in these files in a meaningful way. Fortunately, there's already some functionality using `piel` in this context:

calibration_propagation_delay_sweep_data = pe.types.PropagationDelayMeasurementSweep(
    measurements=calibration_propagation_data,
)
pcb_propagation_delay_sweep_data = pe.types.PropagationDelayMeasurementSweep(
    measurements=pcb_propagation_data, name="frequency_sweep"
)

# So these measurements are just the measurement definition, but do not contain the data. We need to extract it from the files.

calibration_propagation_delay_sweep_data = (
    pe.DPO73304.extract_propagation_delay_measurement_sweep_data(
        calibration_propagation_delay_sweep_data
    )
)
pcb_propagation_delay_sweep_data = (
    pe.DPO73304.extract_propagation_delay_measurement_sweep_data(
        pcb_propagation_delay_sweep_data
    )
)

# Now, we want to plot this files as a function of the sweep parameter. Fortunately this is pretty easy:


fig, ax = pe.visual.plot_signal_propagation_sweep_measurement(
    pcb_propagation_delay_sweep_data,
    measurement_name="delay_ch1_ch2__s_2",
)

fig, ax = pe.visual.plot_signal_propagation_sweep_measurement(
    pcb_propagation_delay_sweep_data,
    measurement_name="delay_ch1_ch2__s_1",
)

# TODO add graph showing this exact setup.
#
# In this setup, we will use a RF signal generator and a RF oscilloscope.
#
# First, we will split the signal generator signal through two paths and see them in the oscillscope. They should overlap each other perfectly. Both signals are terminated at the oscilloscope inputs in order to get an exact rising edge.
