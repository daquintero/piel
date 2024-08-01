from .types import ExperimentInstance
from .types.measurements.map import configuration_to_measurement_map


def compose_measurement_from_experiment_instance(
    experiment_instance: ExperimentInstance,
    configuration_measurement_map: dict = configuration_to_measurement_map,
):
    """
    This function is meant to be run after the measurements have been collected and the data exists within the measurement directories.
    Each experiment instance should correspond to a list of measurement configurations, ie the specific set of measurements
     that are required at each directory generated for the experimental instance. Hence, for this function to work properly,
     it is required to have a mapping between experiment configuration types and measurement classes accordingly.
     The mapping will be between a given MeasurementConfiguration type and a Measurement class which has the references
     of the data containers accordingly.
    """
