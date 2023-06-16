def single_parameter_sweep(
        base_design_configuration: dict,
        parameter_name:str,
        parameter_sweep_values:list,
):
    parameter_sweep_design_dictionary_array = []
    for parameter in parameter_sweep_values:
        design = base_design_configuration.copy()
        design[parameter_name] = parameter
        parameter_sweep_design_dictionary_array.extend([design])
    return parameter_sweep_design_dictionary_array


def multi_parameter_sweep(
        base_design_configuration: dict,
        parameter_sweep_dictionary: dict
):
    """
    This multiparameter sweep is pretty cool, as it will generate designer list of dictionaries that comprise of all the possible combinations of your parameter sweeps. For example, if you are sweeping parameter_1 = np.arange(0, 2) = array([0, 1]), and parameter_2 = np.arange(2, 4) = array([2, 3]), then this function will generate list of dictionaries based on the default_design dictionary, but that will comprise of all the potential parameter combinations within this list.

    For the example above, there arould be 4 combinations [(0, 2), (0, 3), (1, 2), (1, 3)].

    If you were instead sweeping for parameter_1 = np.arange(0, 5) and parameter_2 = np.arange(0, 5), the dictionary generated would correspond to these parameter combinations of [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)].

    Make sure to use the parameter_names from default_design when writing up the parameter_sweep dictionary key name.

    Example input formats::

        example_parameter_sweep_dictionary = {
            "parameter_1": np.arange(1, -40, 1),
            "parameter_2": np.arange(1, -40, 1),
        }

        example_base_design_configuration = {
            "parameter_1": 10.0,
            "parameter_2": 40.0,
            "parameter_3": 0
        }
    """
    parameter_names_sweep_list = []
    parameter_sweep_values_list = []
    parameter_sweep_design_dictionary_array = []

    for parameter_sweep_name in parameter_sweep_dictionary.keys():
        parameter_names_sweep_list.extend([parameter_sweep_name])
        parameter_sweep_values_list.extend([parameter_sweep_dictionary[parameter_sweep_name].tolist()])
        # print(parameter_sweep_values_list)
    sweep_combinations = list(itertools.product(*parameter_sweep_values_list))
    # print(sweep_combinations)
    # print(parameter_names_sweep_list)

    for parameter_combination in sweep_combinations:
        design = base_design_configuration.copy()
        parameter_index = 0

        for parameter in parameter_combination:
            design[parameter_names_sweep_list[parameter_index]] = parameter
            parameter_index += 1
        parameter_sweep_design_dictionary_array.extend([design])

    return parameter_sweep_design_dictionary_array

__all__ = [
    "single_parameter_sweep",
    "multi_parameter_sweep",
]
