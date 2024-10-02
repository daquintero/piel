from __future__ import annotations
from piel.types.connectivity.core import Instance
from ...experiment import Experiment
from .generic import MeasurementDataCollectionTypes
from piel.base.experimental.measurements.data.experiment import index_experiment_data


class ExperimentData(Instance):
    """
    This is a definition of a collection of measurement data.
    """

    experiment: Experiment = None
    data: MeasurementDataCollectionTypes = None

    # TODO add validators to make sure data and measurement parameters are the same size
    # TODO add validators for data type matches experiment type
    __getitem__ = index_experiment_data


class ExperimentDataCollection(Instance):
    collection: list[ExperimentData] = []
