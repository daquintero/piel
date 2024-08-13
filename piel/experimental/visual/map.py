import inspect
from piel.experimental.visual import propagation, frequency


def auto_function_list_from_module(module) -> list[callable]:
    # Get a list of all callable functions defined in the module
    functions = [
        getattr(module, name)
        for name in dir(module)
        if inspect.isfunction(getattr(module, name))
        and getattr(module, name).__module__ == module.__name__
    ]
    return functions


def auto_function_name_list_from_module(module) -> list[str]:
    # Extract all function names defined in the module
    functions_list = [
        name
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    ]

    # Remove "plot_" prefix from the function names
    function_name_list = [name.replace("plot_", "") for name in functions_list]

    # Ensure each function name length is limited to a reasonable length
    # while still ensuring the total file name is under 100 characters
    truncated_function_name_list = [name[:20] for name in function_name_list]

    return truncated_function_name_list


"""
This mapping creates an automatic relationships between the corresponding measurement data and the list of plots
that should be generated for it.
"""
measurement_data_to_plot_map = {
    "PropagationDelayMeasurementData": auto_function_list_from_module(
        propagation.measurement_data
    ),
    "VNASParameterMeasurementData": auto_function_list_from_module(
        frequency.measurement_data
    ),
}

"""
This mapping creates an automatic relationship between the data collection and the plotting required.
"""
measurement_data_collection_to_plot_map = {
    "PropagationDelayMeasurementDataCollection": auto_function_list_from_module(
        propagation.measurement_data_collection
    ),
    "VNASParameterMeasurementDataCollection": auto_function_list_from_module(
        frequency.measurement_data_collection
    ),
}


measurement_data_collection_to_plot_suffix_map = {
    "PropagationDelayMeasurementDataCollection": auto_function_name_list_from_module(
        propagation.measurement_data_collection
    ),
    "VNASParameterMeasurementDataCollection": auto_function_name_list_from_module(
        frequency.measurement_data_collection
    ),
}
