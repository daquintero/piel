from typing import Optional
from ...types import Instance, ComponentTypes, ConnectionTypes
from .measurements.generic import MeasurementConfigurationTypes


class ExperimentInstance(Instance):
    """
    This is a specific instance of an experimental configuration. It can be the result of serialization a given
    experiment configuration. The experiment contains all devices that correspond a specific setup. Each device
    contains a particular configuration, or may have information about the environment, etc.
    """

    components: list[ComponentTypes] | tuple[ComponentTypes]
    """
    Contains all references to the instantiated devices.
    """

    connections: list[ConnectionTypes] | tuple[ConnectionTypes]
    """
    All the connectivity information is stored here.
    """

    goal: Optional[str] = None
    """
    The goal of the experiment test.
    """

    parameters: Optional[dict] = None
    """
    A dictionary of reference parameters in this experimental instance. Does not contain all the serialised experimental data.
    """

    index: Optional[int] = None
    """
    A defined index of the experiment instance tuple.
    """

    date_configured: Optional[str] = None
    """
    The date the experiment was configured.
    """

    date_measured: Optional[str] = None
    """
    The date the experiment was measured.
    """

    measurement_configuration_list: list[MeasurementConfigurationTypes] = None


class Experiment(Instance):
    name: str
    """
    Every experiment is required to have a name.
    """

    goal: Optional[str] = None
    """
    The goal of the complete experiment.
    """

    experiment_instances: list[ExperimentInstance] | tuple[ExperimentInstance]
    """
    Contains all the experiment instances.
    """

    parameters_list: list[dict] = None
    """
    List of basic important parameters in dictionaries used to do basic metadata analysis of the experiment.
    """
