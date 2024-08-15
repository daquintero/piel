# TODO depends on the simulation tool refactor.
# import pytest
# from unittest.mock import MagicMock, patch
# from piel.experimental import (
#     create_experiment_data_collection_from_unique_parameters,
# )
# from piel.experimental.types import (
#     Experiment,
#     ExperimentData,
#     ExperimentDataCollection,
# )
# from piel.models import load_from_dict
# from piel.experimental.measurements.map import measurement_data_to_measurement_collection_data_map
# from piel.utils.parametric import get_unique_dataframe_subsets
#
# def test_create_experiment_data_collection_from_unique_parameters():
#     # Mock the experiment and experiment data
#     mock_experiment_instances = [MagicMock(), MagicMock()]
#     mock_parameters_df = MagicMock()
#     mock_experiment = MagicMock(spec=Experiment)
#     mock_experiment.name = "TestExperiment"
#     mock_experiment.goal = "TestGoal"
#     mock_experiment.experiment_instances = mock_experiment_instances
#     mock_experiment.parameters = mock_parameters_df
#
#     mock_data_collection = MagicMock()
#     mock_experiment_data = MagicMock(spec=ExperimentData)
#     mock_experiment_data.name = "TestExperimentData"
#     mock_experiment_data.experiment = mock_experiment
#     mock_experiment_data.data = {"collection": [mock_data_collection]}
#
#     # Mock the unique dataframe subsets
#     mock_unique_parameter_subset_dictionary = {
#         0: MagicMock(),
#         1: MagicMock(),
#     }
#
#     with patch("piel.utils.parametric.get_unique_dataframe_subsets", return_value=mock_unique_parameter_subset_dictionary), \
#          patch("piel.models.load_from_dict", side_effect=lambda data, model: data if model == ExperimentData else mock_experiment), \
#          patch("piel.experimental.measurements.map.measurement_data_to_measurement_collection_data_map", return_value=MagicMock()):
#
#         result = create_experiment_data_collection_from_unique_parameters(mock_experiment_data)
#
#         # Assertions to check that the function works as expected
#         assert isinstance(result, ExperimentDataCollection)
#         assert len(result.collection) == len(mock_unique_parameter_subset_dictionary)
#
#         for experiment_data_i in result.collection:
#             assert isinstance(experiment_data_i, ExperimentData)
#             assert experiment_data_i.experiment.name.startswith("TestExperiment_")
#             assert experiment_data_i.experiment.goal.startswith("TestGoal_")
#             assert experiment_data_i.data.name.startswith(experiment_data_i.experiment.name)
#
#
# def test_create_experiment_data_collection_handles_empty_experiment():
#     mock_experiment = MagicMock(spec=Experiment)
#     mock_experiment.name = "EmptyExperiment"
#     mock_experiment.goal = "EmptyGoal"
#     mock_experiment.experiment_instances = []
#     mock_experiment.parameters = MagicMock()
#
#     mock_experiment_data = MagicMock(spec=ExperimentData)
#     mock_experiment_data.name = "EmptyExperimentData"
#     mock_experiment_data.experiment = mock_experiment
#     mock_experiment_data.data = {"collection": []}
#
#     mock_unique_parameter_subset_dictionary = {}
#
#     with patch("piel.utils.parametric.get_unique_dataframe_subsets", return_value=mock_unique_parameter_subset_dictionary), \
#          patch("piel.models.load_from_dict", side_effect=lambda data, model: data if model == ExperimentData else mock_experiment):
#
#         result = create_experiment_data_collection_from_unique_parameters(mock_experiment_data)
#
#         assert isinstance(result, ExperimentDataCollection)
#         assert len(result.collection) == 0
#
#
# def test_create_experiment_data_collection_invalid_data_type():
#     mock_experiment = MagicMock(spec=Experiment)
#     mock_experiment_data = MagicMock(spec=ExperimentData)
#     mock_experiment_data.experiment = mock_experiment
#     mock_experiment_data.data = {"collection": [MagicMock()]}
#
#     with patch("piel.utils.parametric.get_unique_dataframe_subsets", return_value={}), \
#          patch("piel.models.load_from_dict", side_effect=lambda data, model: data if model == ExperimentData else mock_experiment), \
#          patch("piel.experimental.measurements.map.measurement_data_to_measurement_collection_data_map", return_value=None):
#
#         with pytest.raises(KeyError):
#             create_experiment_data_collection_from_unique_parameters(mock_experiment_data)
#
#
# # Add more tests as needed to cover additional scenarios and edge cases.
