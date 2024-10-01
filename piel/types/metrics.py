import pandas as pd
from piel.types.connectivity.abstract import Instance
from piel.types.core import NumericalTypes
from piel.types.units import Unit, ratio


class ScalarMetrics(Instance):
    """
    Standard definition for a scalar metrics. It includes the value, mean, min, max, standard deviation and count.
    """

    value: NumericalTypes | None = None
    mean: NumericalTypes | None = None
    min: NumericalTypes | None = None
    max: NumericalTypes | None = None
    standard_deviation: NumericalTypes | None = None
    count: NumericalTypes | None = None
    unit: Unit = ratio

    @property
    def data(self):
        # Create a dictionary with the scalar metrics
        data = {
            "Metric": ["Value", "Mean", "Min", "Max", "Standard Deviation", "Count"],
            "Value": [
                self.value,
                self.mean,
                self.min,
                self.max,
                self.standard_deviation,
                self.count,
            ],
        }
        return data

    @property
    def table(self):
        # Convert to a pandas DataFrame
        df = pd.DataFrame(self.data)
        return df


class ScalarMetricCollection(Instance):
    """
    A collection of scalar metrics useful when analyzing multiple aspects of a design.
    """

    metrics: list[ScalarMetrics] = []

    @property
    def data(self):
        data = dict()
        for metric in self.metrics:
            data[metric.name] = {
                "value": metric.value,
                "mean": metric.mean,
                "min": metric.min,
                "max": metric.max,
                "standard_deviation": metric.standard_deviation,
                "count": metric.count,
                "unit": metric.unit.label,
            }
        return data

    @property
    def table(self):
        """
        Composes a full table with the names and all the individual metrics that are part of this collection.

        Returns:
            pd.DataFrame: A DataFrame containing all scalar metrics with their respective values.
        """
        # Initialize a list to collect metric data
        metrics_data = []

        for metric in self.metrics:
            metrics_data.append(
                {
                    "Name": metric.name,
                    "Value": metric.value,
                    "Mean": metric.mean,
                    "Min": metric.min,
                    "Max": metric.max,
                    "Standard Deviation": metric.standard_deviation,
                    "Count": metric.count,
                    "Unit": metric.unit.label,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(metrics_data)

        # Optional: Set 'Name' as the index for better readability
        df.set_index("Name", inplace=True)

        return df

    def __getitem__(self, index):
        """
        Allows for indexing and slicing of the metrics list.

        Args:
            index: An integer or slice object for indexing.

        Returns:
            A new ScalarMetricCollection containing the specified slice of metrics.
        """
        if isinstance(index, int):
            # Return a new collection with a single metric if indexed by an integer
            metrics = [self.metrics[index]]
        elif isinstance(index, slice):
            # Return a new collection with a slice of the metrics list
            metrics = self.metrics[index]
        else:
            raise TypeError("Invalid index type. Must be int or slice.")

        return ScalarMetricCollection(name=self.name, metrics=metrics)
