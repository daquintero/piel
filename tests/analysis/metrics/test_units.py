# File: test_unit_conversion.py

import pytest
import numpy as np

# Import the functions to test
from piel.analysis.metrics import (
    convert_scalar_metric_unit,
    convert_metric_collection_units_per_metric,
    convert_metric_collection_per_unit,
)

# Import necessary classes and units
from piel.types import ScalarMetrics, ScalarMetricCollection, Unit


def create_scalar_metrics(
    name: str,
    value: float,
    mean: float,
    min_val: float,
    max_val: float,
    standard_deviation: float,
    count: int,
    unit: Unit,
) -> ScalarMetrics:
    """
    Helper function to create a ScalarMetrics instance.
    """
    return ScalarMetrics(
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


def test_convert_scalar_metric_unit_success():
    """
    Test converting ScalarMetrics unit successfully when units are compatible.
    """
    # Original metric in volts
    original_unit = Unit(name="volt", datum="voltage", base=1, label="V")
    target_unit = Unit(name="millivolt", datum="voltage", base=1e-3, label="mV")

    metric = create_scalar_metrics(
        name="metric1",
        value=10.0,
        mean=10.0,
        min_val=5.0,
        max_val=15.0,
        standard_deviation=2.0,
        count=3,
        unit=original_unit,
    )

    # Call function
    converted_metric = convert_scalar_metric_unit(metric, target_unit)

    # Expected values: multiplied by (1 / 1e-3) = 1000
    assert (
        converted_metric.unit == target_unit
    ), "Unit should be updated to target unit."
    assert (
        converted_metric.value == 10.0 * (original_unit.base / target_unit.base)
    ), f"Expected value={10.0 * (original_unit.base / target_unit.base)}, got {converted_metric.value}"
    assert (
        converted_metric.mean == 10.0 * (original_unit.base / target_unit.base)
    ), f"Expected mean={10.0 * (original_unit.base / target_unit.base)}, got {converted_metric.mean}"
    assert (
        converted_metric.min == 5.0 * (original_unit.base / target_unit.base)
    ), f"Expected min={5.0 * (original_unit.base / target_unit.base)}, got {converted_metric.min}"
    assert (
        converted_metric.max == 15.0 * (original_unit.base / target_unit.base)
    ), f"Expected max={15.0 * (original_unit.base / target_unit.base)}, got {converted_metric.max}"
    assert (
        converted_metric.standard_deviation
        == 2.0 * (original_unit.base / target_unit.base)
    ), f"Expected std_dev={2.0 * (original_unit.base / target_unit.base)}, got {converted_metric.standard_deviation}"
    assert converted_metric.count == 3, "Count should remain the same."


def test_convert_scalar_metric_unit_incompatible_units():
    """
    Test converting ScalarMetrics unit raises ValueError when units are incompatible.
    """
    # Original metric in volts
    original_unit = Unit(name="volt", datum="voltage", base=1, label="V")
    target_unit = Unit(name="ampere", datum="ampere", base=1, label="A")

    metric = create_scalar_metrics(
        name="metric1",
        value=10.0,
        mean=10.0,
        min_val=5.0,
        max_val=15.0,
        standard_deviation=2.0,
        count=3,
        unit=original_unit,
    )

    # Call function and expect ValueError
    with pytest.raises(
        ValueError,
        match="Cannot convert from unit 'volt' \(datum: voltage\) to unit 'ampere' \(datum: ampere\). Units are incompatible.",
    ):
        convert_scalar_metric_unit(metric, target_unit)


def test_convert_scalar_metric_unit_none_values():
    """
    Test converting ScalarMetrics unit when some numerical fields are None.
    """
    # Original metric in volts
    original_unit = Unit(name="volt", datum="voltage", base=1, label="V")
    target_unit = Unit(name="kilovolt", datum="voltage", base=1e3, label="kV")

    metric = create_scalar_metrics(
        name="metric1",
        value=None,
        mean=None,
        min_val=5.0,
        max_val=15.0,
        standard_deviation=None,
        count=3,
        unit=original_unit,
    )

    # Call function
    converted_metric = convert_scalar_metric_unit(metric, target_unit)

    # Expected: value and mean remain None; min and max are multiplied; std_dev remains None
    assert (
        converted_metric.unit == target_unit
    ), "Unit should be updated to target unit."
    assert converted_metric.value is None, "Value should remain None."
    assert converted_metric.mean is None, "Mean should remain None."
    assert (
        converted_metric.min == 5.0 * (original_unit.base / target_unit.base)
    ), f"Expected min={5.0 * (original_unit.base / target_unit.base)}, got {converted_metric.min}"
    assert (
        converted_metric.max == 15.0 * (original_unit.base / target_unit.base)
    ), f"Expected max={15.0 * (original_unit.base / target_unit.base)}, got {converted_metric.max}"
    assert (
        converted_metric.standard_deviation is None
    ), "Standard deviation should remain None."
    assert converted_metric.count == 3, "Count should remain the same."


def test_convert_metric_collection_units_per_metric_success():
    """
    Test converting units of a ScalarMetricCollection using metric names.
    """
    # Create metrics
    metric1 = create_scalar_metrics(
        "voltage_metric",
        10.0,
        10.0,
        5.0,
        15.0,
        2.0,
        3,
        Unit(name="volt", datum="voltage", base=1, label="V"),
    )
    metric2 = create_scalar_metrics(
        "current_metric",
        20.0,
        20.0,
        15.0,
        25.0,
        3.0,
        5,
        Unit(name="ampere", datum="ampere", base=1, label="A"),
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1, metric2])

    # Define target units per metric name
    target_units = {
        "voltage_metric": Unit(
            name="millivolt", datum="voltage", base=1e-3, label="mV"
        ),
        "current_metric": Unit(
            name="milliampere", datum="ampere", base=1e-3, label="mA"
        ),
    }

    # Call function
    converted_collection = convert_metric_collection_units_per_metric(
        collection, target_units
    )

    # Assertions for metric1
    converted_metric1 = converted_collection.metrics[0]
    assert (
        converted_metric1.name == "voltage_metric"
    ), "Metric1 name should remain the same."
    assert (
        converted_metric1.unit == target_units["voltage_metric"]
    ), "Metric1 unit should be converted to millivolt."
    assert (
        converted_metric1.value
        == 10.0 * (metric1.unit.base / target_units["voltage_metric"].base)
    ), f"Expected value={10.0 * (metric1.unit.base / target_units['voltage_metric'].base)}, got {converted_metric1.value}"
    assert (
        converted_metric1.mean
        == 10.0 * (metric1.unit.base / target_units["voltage_metric"].base)
    ), f"Expected mean={10.0 * (metric1.unit.base / target_units['voltage_metric'].base)}, got {converted_metric1.mean}"
    assert (
        converted_metric1.min
        == 5.0 * (metric1.unit.base / target_units["voltage_metric"].base)
    ), f"Expected min={5.0 * (metric1.unit.base / target_units['voltage_metric'].base)}, got {converted_metric1.min}"
    assert (
        converted_metric1.max
        == 15.0 * (metric1.unit.base / target_units["voltage_metric"].base)
    ), f"Expected max={15.0 * (metric1.unit.base / target_units['voltage_metric'].base)}, got {converted_metric1.max}"
    assert (
        converted_metric1.standard_deviation
        == 2.0 * (metric1.unit.base / target_units["voltage_metric"].base)
    ), f"Expected std_dev={2.0 * (metric1.unit.base / target_units['voltage_metric'].base)}, got {converted_metric1.standard_deviation}"

    # Assertions for metric2
    converted_metric2 = converted_collection.metrics[1]
    assert (
        converted_metric2.name == "current_metric"
    ), "Metric2 name should remain the same."
    assert (
        converted_metric2.unit == target_units["current_metric"]
    ), "Metric2 unit should be converted to milliampere."
    assert (
        converted_metric2.value
        == 20.0 * (metric2.unit.base / target_units["current_metric"].base)
    ), f"Expected value={20.0 * (metric2.unit.base / target_units['current_metric'].base)}, got {converted_metric2.value}"
    assert (
        converted_metric2.mean
        == 20.0 * (metric2.unit.base / target_units["current_metric"].base)
    ), f"Expected mean={20.0 * (metric2.unit.base / target_units['current_metric'].base)}, got {converted_metric2.mean}"
    assert (
        converted_metric2.min
        == 15.0 * (metric2.unit.base / target_units["current_metric"].base)
    ), f"Expected min={15.0 * (metric2.unit.base / target_units['current_metric'].base)}, got {converted_metric2.min}"
    assert (
        converted_metric2.max
        == 25.0 * (metric2.unit.base / target_units["current_metric"].base)
    ), f"Expected max={25.0 * (metric2.unit.base / target_units['current_metric'].base)}, got {converted_metric2.max}"
    assert (
        converted_metric2.standard_deviation
        == 3.0 * (metric2.unit.base / target_units["current_metric"].base)
    ), f"Expected std_dev={3.0 * (metric2.unit.base / target_units['current_metric'].base)}, got {converted_metric2.standard_deviation}"


def test_convert_metric_collection_units_per_metric_missing_metric():
    """
    Test converting units of a ScalarMetricCollection when target_units is missing a metric name.
    """
    # Create metrics
    metric1 = create_scalar_metrics(
        "voltage_metric",
        10.0,
        10.0,
        5.0,
        15.0,
        2.0,
        3,
        Unit(name="volt", datum="voltage", base=1, label="V"),
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1])

    # Define target units with missing metric
    target_units = {
        # "voltage_metric" missing
    }

    # Call function and expect ValueError
    with pytest.raises(
        ValueError,
        match="Target unit for metric 'voltage_metric' not provided in target_units dictionary.",
    ):
        convert_metric_collection_units_per_metric(collection, target_units)


def test_convert_metric_collection_units_per_metric_incompatible_units():
    """
    Test converting units of a ScalarMetricCollection when units are incompatible.
    """
    # Create metrics
    metric1 = create_scalar_metrics(
        "voltage_metric",
        10.0,
        10.0,
        5.0,
        15.0,
        2.0,
        3,
        Unit(name="volt", datum="voltage", base=1, label="V"),
    )

    # Create collection
    collection = create_scalar_metric_collection([metric1])

    # Define target units with incompatible datum
    target_units = {
        "voltage_metric": Unit(name="ampere", datum="ampere", base=1, label="A"),
    }

    # Call function and expect ValueError
    with pytest.raises(
        ValueError,
        match="Cannot convert from unit 'volt' .* to unit 'ampere' .* Units are incompatible.",
    ):
        convert_metric_collection_units_per_metric(collection, target_units)
