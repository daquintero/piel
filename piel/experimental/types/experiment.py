from ...types import Instance, ComponentTypes, ConnectionTypes, PathTypes
from .measurements.generic import MeasurementConfigurationTypes


class ExperimentInstance(Instance):
    """
    This is a specific instance of an experimental configuration. It can be the result of serialization a given
    experiment configuration. The experiment contains all devices that correspond a specific setup. Each device
    contains a particular configuration, or may have information about the environment, etc.
    """

    components: list[ComponentTypes] | tuple[ComponentTypes] = []
    """
    Contains all references to the instantiated devices.
    """

    connections: list[ConnectionTypes] | tuple[ConnectionTypes] = []
    """
    All the connectivity information is stored here.
    """

    goal: str = ""
    """
    The goal of the experiment test.
    """

    parameters: dict = {}
    """
    A dictionary of reference parameters in this experimental instance. Does not contain all the serialised experimental data.
    """

    index: int = 0
    """
    A defined index of the experiment instance tuple.
    """

    date_configured: str = ""
    """
    The date the experiment was configured.
    """

    date_measured: str = ""
    """
    The date the experiment was measured.
    """

    measurement_configuration_list: list[MeasurementConfigurationTypes] = []


class Experiment(Instance):
    name: str = ""
    """
    Every experiment is required to have a name.
    """

    goal: str = ""
    """
    The goal of the complete experiment.
    """

    experiment_instances: list[ExperimentInstance] = []
    """
    Contains all the experiment instances.
    """

    parameters_list: list[dict] = [{}]
    """
    List of basic important parameters in dictionaries used to do basic metadata analysis of the experiment.
    """

    parent_directory: PathTypes = ""
    """
    Optional parameter to specify the `parent_directory` of the Experiment, where the directories containing the data
    and metadata of the `ExperimentInstances` are constructed.
    """

    @property
    def parameters(self):
        import pandas as pd  # TODO maybe move this?

        return pd.DataFrame(self.parameters_list)


class ExperimentCollection(Instance):
    """
    This class contains a collection of different experiments,
    each which may contain a large set of measurements internally.
    """

    experiment_list: list[Experiment] = []
