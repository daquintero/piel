# File: test_rename_metrics_collection.py

import pytest

# Import the function to test
from piel.analysis.metrics import rename_metrics_collection

# Import necessary classes and units
from piel.types import ScalarMetricCollection, ScalarMetric, Unit

# Sample Units
VOLTAGE_UNIT = Unit(name="volt", datum="voltage", base=1, label="V")
CURRENT_UNIT = Unit(name="ampere", datum="ampere", base=1, label="A")


def create_scalar_metrics(
    name: str,
    value: float,
    mean: float,
    min_val: float,
    max_val: float,
    standard_deviation: float,
    count: int,
    unit: Unit,
) -> ScalarMetric:
    """
    Helper function to create a ScalarMetric instance.
    """
    return ScalarMetric(
        name=name,
        value=value,
        mean=mean,
        min=min_val,
        max=max_val,
        standard_deviation=standard_deviation,
        count=count,
        unit=unit,
    )


def create_scalar_metric_collection(metrics: list) -> ScalarMetricCollection:
    """
    Helper function to create a ScalarMetricCollection instance.
    """
    return ScalarMetricCollection(metrics=metrics)


def test_rename_metrics_collection_success():
    """
    Test renaming metrics collection with correct number of new names.
    """
    # Create sample metrics
    metric1 = create_scalar_metrics(
        "metric1", 10.0, 10.0, 5.0, 15.0, 2.0, 3, VOLTAGE_UNIT
    )
    metric2 = create_scalar_metrics(
        "metric2", 20.0, 20.0, 15.0, 25.0, 3.0, 5, CURRENT_UNIT
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1, metric2])

    # New names
    new_names = ["new_metric1", "new_metric2"]

    # Call function
    new_collection = rename_metrics_collection(collection, new_names)

    # Assertions
    assert len(new_collection.metrics) == 2, "Number of metrics should remain the same."
    assert (
        new_collection.metrics[0].name == "new_metric1"
    ), "First metric should be renamed."
    assert (
        new_collection.metrics[1].name == "new_metric2"
    ), "Second metric should be renamed."
    # Ensure original collection is unchanged if copy is used
    assert (
        collection.metrics[0].name == "metric1"
    ), "Original collection should remain unchanged."
    assert (
        collection.metrics[1].name == "metric2"
    ), "Original collection should remain unchanged."


def test_rename_metrics_collection_incorrect_new_names_length():
    """
    Test renaming metrics collection with incorrect number of new names.
    """
    # Create sample metrics
    metric1 = create_scalar_metrics(
        "metric1", 10.0, 10.0, 5.0, 15.0, 2.0, 3, VOLTAGE_UNIT
    )
    metric2 = create_scalar_metrics(
        "metric2", 20.0, 20.0, 15.0, 25.0, 3.0, 5, CURRENT_UNIT
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1, metric2])

    # New names with incorrect length
    new_names = ["new_metric1"]  # Only one name for two metrics

    # Call function and expect ValueError
    with pytest.raises(
        ValueError,
        match="Number of new names \\(1\\) does not match the number of metrics in the collection \\(2\\).",
    ):
        rename_metrics_collection(collection, new_names)


def test_rename_metrics_collection_empty_metrics():
    """
    Test renaming metrics collection when the collection has no metrics.
    """
    # Create empty collection
    collection = create_scalar_metric_collection([])

    # New names (empty)
    new_names = []

    # Call function
    new_collection = rename_metrics_collection(collection, new_names)

    # Assertions
    assert (
        len(new_collection.metrics) == 0
    ), "Renamed collection should also have no metrics."


def test_rename_metrics_collection_duplicate_new_names():
    """
    Test renaming metrics collection with duplicate new names.
    """
    # Create sample metrics
    metric1 = create_scalar_metrics(
        "metric1", 10.0, 10.0, 5.0, 15.0, 2.0, 3, VOLTAGE_UNIT
    )
    metric2 = create_scalar_metrics(
        "metric2", 20.0, 20.0, 15.0, 25.0, 3.0, 5, CURRENT_UNIT
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1, metric2])

    # New names with duplicates
    new_names = ["duplicate_name", "duplicate_name"]

    # Call function
    new_collection = rename_metrics_collection(collection, new_names)

    # Assertions
    assert len(new_collection.metrics) == 2, "Number of metrics should remain the same."
    assert (
        new_collection.metrics[0].name == "duplicate_name"
    ), "First metric should be renamed."
    assert (
        new_collection.metrics[1].name == "duplicate_name"
    ), "Second metric should be renamed."


def test_rename_metrics_collection_non_string_names():
    """
    Test renaming metrics collection with non-string new names.
    """
    # Create sample metrics
    metric1 = create_scalar_metrics(
        "metric1", 10.0, 10.0, 5.0, 15.0, 2.0, 3, VOLTAGE_UNIT
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1])

    # New names with non-string
    new_names = [123]  # Non-string name
