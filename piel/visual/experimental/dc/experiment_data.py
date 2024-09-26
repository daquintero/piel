from piel.types.experimental import ExperimentData
from . import measurement_data_collection


def plot_two_port_dc_sweep(experiment_data: ExperimentData, **kwargs) -> tuple:
    # TODO Implement validation that it's a time-propagation delay measurement
    label_list = experiment_data.experiment.parameters.values[:, 0]
    return measurement_data_collection.plot_two_port_dc_sweep(
        dc_sweep_collection=experiment_data.data, label_list=label_list, **kwargs
    )
