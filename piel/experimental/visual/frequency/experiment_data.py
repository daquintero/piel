from ...types import ExperimentData
from . import measurement_data_collection


def plot_s_parameter_real_and_imaginary(
    experiment_data: ExperimentData,
    **kwargs,
) -> tuple:
    data_collection = experiment_data.data

    measurement_data_collection.plot_s_parameter_real_and_imaginary(
        data_collection=data_collection, **kwargs
    )
