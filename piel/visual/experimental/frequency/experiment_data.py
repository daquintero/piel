from piel.types.experimental import ExperimentData
from . import measurement_data_collection


def plot_s_parameter_real_and_imaginary(
    experiment_data: ExperimentData,
    **kwargs,
) -> tuple:
    data_collection = experiment_data.data

    parameters_list = experiment_data.experiment.parameters_list

    measurement_data_collection.plot_s_parameter_real_and_imaginary(
        data_collection=data_collection, parameters_list=parameters_list, **kwargs
    )
