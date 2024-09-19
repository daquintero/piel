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
    def table(self):
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
        # Convert to a pandas DataFrame
        df = pd.DataFrame(data)
        return df
