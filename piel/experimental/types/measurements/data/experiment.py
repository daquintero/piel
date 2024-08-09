from .....types import Instance
from ...experiment import Experiment
from .generic import MeasurementDataCollectionTypes


class ExperimentData(Instance):
    """
    This is a definition of a collection of experimental data.
    """

    experiment: Experiment
    data: MeasurementDataCollectionTypes

    # TODO add validators to make sure data and experimental parameters are the same size
    # TODO add validators for data type matches experiment type

    @property
    def parameter_map(self):
        """
        This function creates a dictionary between the experiment.parameters
        and the corresponding data instance so that it's easier to index accordingly.
        """
        # TODO finish
        # import pandas as pd
        # data_dictionary = {"data": data_i for data_i in data}
        # print(data_dictionary)
        # dataframe = pd.DataFrame(self.experiment.parameters_list)
