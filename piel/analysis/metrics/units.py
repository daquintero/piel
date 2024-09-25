from piel.types import Unit, ScalarMetrics, ScalarMetricCollection

from pydantic import ValidationError


def convert_scalar_metric_unit(
    metric: ScalarMetrics, target_unit: Unit
) -> ScalarMetrics:
    """
    Converts the units of a single ScalarMetrics instance to the target unit.

    Args:
        metric (ScalarMetrics): The original scalar metric.
        target_unit (Unit): The target unit to convert to.

    Returns:
        ScalarMetrics: A new ScalarMetrics instance with converted values and updated unit.

    Raises:
        ValueError: If the original unit and target unit have different 'datum'.
    """
    original_unit = metric.unit
    if original_unit.datum != target_unit.datum:
        raise ValueError(
            f"Cannot convert from unit '{original_unit.name}' (datum: {original_unit.datum}) "
            f"to unit '{target_unit.name}' (datum: {target_unit.datum}). Units are incompatible."
        )

    # Calculate conversion factor
    conversion_factor = original_unit.base / target_unit.base

    # Define a helper function to convert individual numerical fields
    def convert_field(value):
        if value is None:
            return None
        return value * conversion_factor

    # Create a new ScalarMetrics instance with converted values
    try:
        converted_metric = metric.model_copy(
            update={
                "value": convert_field(metric.value),
                "mean": convert_field(metric.mean),
                "min": convert_field(metric.min),
                "max": convert_field(metric.max),
                "standard_deviation": convert_field(metric.standard_deviation),
                "unit": target_unit,
            }
        )
    except ValidationError as e:
        raise ValueError(f"Error during conversion: {e}")

    return converted_metric


def convert_metric_collection_units_per_metric(
    collection: ScalarMetricCollection, target_units: dict[str, Unit]
) -> ScalarMetricCollection:
    """
    Converts the units of metrics in a ScalarMetricCollection to the target units.

    Args:
        collection (ScalarMetricCollection): The original metric collection.
        target_units (dict[str, Unit]):
            - If a dictionary is provided, keys should be metrics names and values are the target Units.

    Returns:
        ScalarMetricCollection: A new ScalarMetricCollection with converted metrics.

    Raises:
        ValueError: If target_units is a dict and a metric name is missing,
                    or if any unit conversion is invalid.
    """
    converted_metrics = []

    if isinstance(target_units, dict):
        # Convert using a mapping of metric names to target units
        for metric in collection.metrics:
            if metric.name not in target_units:
                raise ValueError(
                    f"Target unit for metric '{metric.name}' not provided in target_units dictionary."
                )
            target_unit = target_units[metric.name]
            converted_metric = convert_scalar_metric_unit(metric, target_unit)
            converted_metrics.append(converted_metric)

    # Create a new ScalarMetricCollection with the converted metrics
    try:
        converted_collection = collection.model_copy(
            update={"metrics": converted_metrics}
        )
    except ValidationError as e:
        raise ValueError(f"Error during collection conversion: {e}")

    return converted_collection


def convert_metric_collection_per_unit(
    collection: ScalarMetricCollection, target_units: dict[str, Unit]
) -> ScalarMetricCollection:
    """
    Converts the units of metrics in a ScalarMetricCollection based on unit names.

    Args:
        collection (ScalarMetricCollection): The original metric collection.
        target_units (dict[str, Unit] ):

    Returns:
        ScalarMetricCollection: A new ScalarMetricCollection with converted metrics.

    Raises:
        ValueError: If target_units is a dict and a metric's unit name is missing,
                    or if any unit conversion is invalid.
    """
    converted_metrics = []

    if isinstance(target_units, dict):
        # Convert using a mapping of unit names to target units
        for metric in collection.metrics:
            current_unit_name = metric.unit.name
            if current_unit_name not in target_units:
                # If the metric's unit is not in the mapping, keep it unchanged
                converted_metrics.append(metric)
                continue
            target_unit = target_units[current_unit_name]
            try:
                converted_metric = convert_scalar_metric_unit(metric, target_unit)
                converted_metrics.append(converted_metric)
            except ValueError as ve:
                raise ValueError(
                    f"Error converting metric '{metric.name}': {ve}"
                ) from ve

    # Create a new ScalarMetricCollection with the converted metrics
    try:
        converted_collection = collection.model_copy(
            update={"metrics": converted_metrics}
        )
    except ValidationError as e:
        raise ValueError(f"Error during collection conversion: {e}") from e

    return converted_collection
