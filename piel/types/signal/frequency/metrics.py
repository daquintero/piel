from piel.types.core import PielBaseModel
from piel.types.metrics import ScalarMetric
from piel.types.signal.frequency.core import Phasor


class FrequencyMetric(PielBaseModel):
    """
    Creates a mapping between a given input and a given scalar metrics response.

    This assumes the following relationships structure:
        Network(INPUT) = OUTPUT

    This means that there is a directional relationship between the input and the transmission output. This type
    of metric includes this dimensionality assuming the metric is always related to the transmission output and
    that the input can be represented in the frequency domain as input. However, for the sake of completeness the
    input will just be called input, in case there is a higher dimensionality input into a frequency network.

    This may still be valid also for other DC networks and inputs in higher-dimensional relationships as a given
    response might be a result of multiple values within a higher level system.
    """

    input: Phasor | None = None
    metric: ScalarMetric = ScalarMetric()

    @property
    def table(self):
        import pandas as pd

        # Initialize an empty list to hold DataFrames
        tables = []

        # If input Phasor exists, process its table
        if self.input:
            phasor_df = self.input.table.copy()
            # Rename columns to match ScalarMetric's table
            # phasor_df = phasor_df.rename(columns={'description': 'metric'})
            tables.append(phasor_df)

        # Add ScalarMetric's table
        scalar_df = self.metric.table.copy()
        tables.append(scalar_df)

        # Concatenate all tables vertically
        combined_df = pd.concat(tables, ignore_index=True)

        return combined_df


class FrequencyMetricCollection(PielBaseModel):
    metrics: list[FrequencyMetric] = []
