from piel.types import ExperimentData


def compose_xarray_dataset_from_experiment_data(
    experiment_data: "ExperimentData",
):
    """
    Composes an xarray.Dataset from an ExperimentData instance, using all experiment parameters as coordinates.

    Args:
        experiment_data (ExperimentData): The experiment data containing parameters and measurements.

    Returns:
        xr.Dataset: An xarray Dataset containing the measurements indexed by all parameters and metric name.

    Raises:
        ValueError: If the number of parameters does not match the number of measurement data entries.
        AttributeError: If any measurement data entry lacks the `measurements` attribute.
    """
    import xarray as xr
    import pandas as pd

    # Extract parameters DataFrame
    parameters_df = experiment_data.experiment.parameters
    measurements_collection = (
        experiment_data.data
    )  # Expected to be PropagationDelayMeasurementDataCollection

    # Validate that the number of parameters matches the number of measurement data entries
    if len(parameters_df) != len(measurements_collection.collection):
        raise ValueError(
            f"Number of parameter entries ({len(parameters_df)}) does not match "
            f"number of measurement data entries ({len(measurements_collection.collection)})."
        )

    # Get list of parameter columns
    parameter_columns = parameters_df.columns.tolist()

    # Initialize a list to hold all records
    data_records = []

    # Iterate over the parameters and corresponding measurement data
    for i, (param_index, param_row) in enumerate(parameters_df.iterrows()):
        # Extract parameter values as a dict
        param_values = param_row.to_dict()

        # Retrieve the corresponding measurement data
        measurement_data_i = measurements_collection.collection[i]

        # Ensure that measurements are available
        if measurement_data_i.measurements is None:
            raise AttributeError(
                f"This function can only compose a dataset when there is a `measurements` "
                f"attribute in the data collection. Measurement data at index {i} has no measurements."
            )

        measurement_table = measurement_data_i.measurements.table

        # Iterate over each row in the measurements table
        for j, row in measurement_table.iterrows():
            # Create a record combining parameter values and measurement data
            record = {**param_values}  # Unpack all parameter columns
            record.update(
                {
                    "metric_name": row.name,  # Assuming 'Name' is set as index
                    "value": row["Value"],
                    "mean": row["Mean"],
                    "min": row["Min"],
                    "max": row["Max"],
                    "standard_deviation": row["Standard Deviation"],
                    "count": row["Count"],
                    "unit": row["Unit"],
                }
            )
            data_records.append(record)

    # Convert the list of records to a DataFrame
    combined_df = pd.DataFrame(data_records)

    # Add a measurement index to handle duplicate metric names per parameter set
    # This ensures each measurement is uniquely identifiable
    # combined_df['measurement_index'] = combined_df.groupby(parameter_columns + ['metric_name']).cumcount()

    # Set multi-index with all parameter columns, metric_name, and measurement_index
    index_columns = parameter_columns + ["metric_name"]
    combined_df.set_index(index_columns, inplace=True)

    # Convert the DataFrame to an xarray.Dataset
    dataset = xr.Dataset.from_dataframe(combined_df)

    return dataset
