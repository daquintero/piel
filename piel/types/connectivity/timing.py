from typing import Optional
from piel.types.core import NumericalTypes
from piel.types.connectivity.core import Instance


class ScalarMetrics(Instance):
    """
    Standard definition for a scalar metrics. It includes the value, mean, min, max, standard deviation and count.
    """

    value: NumericalTypes
    mean: Optional[NumericalTypes]
    min: Optional[NumericalTypes]
    max: Optional[NumericalTypes]
    standard_deviation: Optional[NumericalTypes]


class TimeMetrics(ScalarMetrics):
    """
    This class contains stastical timing information about a given path.

    Notes
    =====

      Each component has some relevant time metrics which characterize
      the propagation of a signal across it. Now, this is more intricate in the case
      of dispersive of multi-frequency signals propagating through a device since
      propagation is dependent on signal frequency. One way to conceptualize it
      is that each frequency corresponds to an individual frequency, but this ignores
      recombination or superposition between the frequencies in a dispersive model.

      Also, timing information is in relation to the input signal that is being propagated through a component.

      We can create a higher level dispersive time metrics for multi-frequency components based on this one.

      The relevant timing we are interested in is as follows, as it has multiple definitions

      - In-to-Out Timing Propagation through a component
      - Group Delay of a signal through an interconnect
      - 10% to 90% rise time of the output waveform of an active component

      Having a shared timing definition is essential to perform correct timing analysis
    """

    value: NumericalTypes = 0
    mean: Optional[NumericalTypes] = 0
    min: Optional[NumericalTypes] = 0
    max: Optional[NumericalTypes] = 0
    standard_deviation: Optional[NumericalTypes] = 0


class DispersiveTimeMetrics(Instance):
    """
    A dispersive time-metrics is useful to represent multi-frequency timing information
    based on the harmonic nature of signals.
    """

    frequency_group: dict[float, TimeMetrics] = {}
    """
  Definition of a mutli-frequency component.
  """


TimeMetricsTypes = TimeMetrics | DispersiveTimeMetrics
"""
Corresponds to all the implemented timing metrics accordingly.
"""

ZeroTimeMetrics = TimeMetrics(value=0, mean=0, min=0, max=0, standard_deviation=0)
"""
Default zero-time metrics defined.
"""
