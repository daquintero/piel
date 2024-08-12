import pandas as pd
import itertools
from itertools import combinations


def single_parameter_sweep(
    base_design_configuration: dict,
    parameter_name: str,
    parameter_sweep_values: list,
):
    """
    This function takes a base_design_configuration dictionary and sweeps a single parameter over a list of values. It returns a list of dictionaries that correspond to the parameter sweep.

    Args:
        base_design_configuration(dict): Base design configuration dictionary.
        parameter_name(str): Name of parameter to sweep.
        parameter_sweep_values(list): List of values to sweep.

    Returns:
        parameter_sweep_design_dictionary_array(list): List of dictionaries that correspond to the parameter sweep.
    """
    parameter_sweep_design_dictionary_array = []
    for parameter in parameter_sweep_values:
        design = base_design_configuration.copy()
        design[parameter_name] = parameter
        parameter_sweep_design_dictionary_array.extend([design])
    return parameter_sweep_design_dictionary_array


def multi_parameter_sweep(
    base_design_configuration: dict, parameter_sweep_dictionary: dict
) -> list:
    """
    This multiparameter sweep is pretty cool, as it will generate designer list of dictionaries that comprise of all the possible combinations of your parameter sweeps. For example, if you are sweeping `parameter_1 = np.arange(0, 2) = array([0, 1])`, and `parameter_2 = np.arange(2, 4) = array([2, 3])`, then this function will generate list of dictionaries based on the default_design dictionary, but that will comprise of all the potential parameter combinations within this list.

    For the example above, there arould be 4 combinations [(0, 2), (0, 3), (1, 2), (1, 3)].

    If you were instead sweeping for `parameter_1 = np.arange(0, 5)` and `parameter_2 = np.arange(0, 5)`, the dictionary generated would correspond to these parameter combinations of::
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)].

    Make sure to use the parameter_names from default_design when writing up the parameter_sweep dictionary key name.

    Example project_structure formats::

        example_parameter_sweep_dictionary = {
            "parameter_1": np.arange(1, -40, 1),
            "parameter_2": np.arange(1, -40, 1),
        }

        example_base_design_configuration = {
            "parameter_1": 10.0,
            "parameter_2": 40.0,
            "parameter_3": 0,
        }

    Args:
        base_design_configuration(dict): Dictionary of the default design configuration.
        parameter_sweep_dictionary(dict): Dictionary of the parameter sweep. The keys should be the same as the keys in the base_design_configuration dictionary.

    Returns:
        parameter_sweep_design_dictionary_array(list): List of dictionaries that comprise of all the possible combinations of your parameter sweeps.
    """
    parameter_names_sweep_list = []
    parameter_sweep_values_list = []
    parameter_sweep_design_dictionary_array = []

    for parameter_sweep_name in parameter_sweep_dictionary.keys():
        parameter_names_sweep_list.extend([parameter_sweep_name])
        parameter_sweep_values_list.extend(
            [parameter_sweep_dictionary[parameter_sweep_name].tolist()]
        )
    sweep_combinations = list(itertools.product(*parameter_sweep_values_list))

    for parameter_combination in sweep_combinations:
        design = base_design_configuration.copy()
        parameter_index = 0
        for parameter in parameter_combination:
            design[parameter_names_sweep_list[parameter_index]] = parameter
            parameter_index += 1
        parameter_sweep_design_dictionary_array.extend([design])

    return parameter_sweep_design_dictionary_array


def get_unique_dataframe_subsets(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    This function takes a pandas DataFrame and returns a list of unique subsets of the DataFrame.
    It is useful for identifying unique operating points in a dataset.
    The function returns a list of DataFrames that contain the unique combinations of values for each column in the input DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    list[str, pd.DataFrame]
        A list of tuples where the first element is a string identifier for the subset and the second element is a DataFrame containing
    """

    # Modified function to create an identifier string with columns and elements together
    def create_identifier(columns, elements):
        """
        This function takes a list of columns and a list of elements and returns a string identifier for the subset.
        """
        return "_".join([f"{col}_{val}" for col, val in zip(columns, elements)])

    subset_dictionary = {}

    for r in range(1, len(dataframe.columns) + 1):
        for column_combination in combinations(dataframe.columns, r):
            # Get the unique combinations of values for the selected columns
            unique_combinations = (
                dataframe[list(column_combination)].drop_duplicates().values
            )

            for unique_value_combination in unique_combinations:
                # Filter rows that match the unique value combination across the selected columns
                subset = dataframe[
                    (
                        dataframe[list(column_combination)] == unique_value_combination
                    ).all(axis=1)
                ]

                # Only consider the subset if there are two or more identical elements
                if len(subset) >= 2:
                    identifier = create_identifier(
                        column_combination, unique_value_combination
                    )
                    subset_dictionary[identifier] = subset.copy()

    return subset_dictionary
