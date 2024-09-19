from piel.types.core import PielBaseModel
from piel.types.metrics import ScalarMetrics
from .connectivity.physical import PhysicalComponent
from typing import Any

FrequencyNetworkModel = Any | None  # TODO temporary nonetype till reserialization
"""
Corresponds to a container that contains a s-parameter representation.

This type alias is currently a placeholder (Any | None) and is intended to be
replaced with a proper s-parameter representation model in the future.

This also represents a frequency, and hence wavelength spectrum if transformed
accordingly.
"""


class FrequencyMetricsCollection(PielBaseModel):
    """
    A collection of frequency-related metrics for RF components.

    Attributes:
    -----------
    bandwidth_Hz : ScalarMetrics
        The bandwidth of the RF component in Hertz.
        Represented as a ScalarMetrics object, which may include
        properties like mean, standard deviation, etc.

    center_transmission_dB : ScalarMetrics
        The center transmission of the RF component in decibels.
        Represented as a ScalarMetrics object, which may include
        properties like mean, standard deviation, etc.
    """

    bandwidth_Hz: ScalarMetrics = ScalarMetrics()
    center_transmission_dB: ScalarMetrics = ScalarMetrics()


class RFPhysicalComponent(PhysicalComponent):
    """
    Represents a physical RF (Radio Frequency) component with frequency-related properties.

    This class extends the PhysicalComponent class to include RF-specific attributes.

    Attributes:
    -----------
    network : FrequencyNetworkModel | None
        A representation of the component's frequency network, typically containing
        s-parameter data. This is currently a placeholder and may be None.

    metrics : FrequencyMetricsCollection
        A collection of frequency-related metrics for this RF component,
        including bandwidth and center transmission.

    Inherits all attributes from PhysicalComponent.

    Notes:
    ------
    - The 'network' attribute is currently using a placeholder type (Any | None)
      and is intended to be updated with a proper s-parameter representation in the future.
    - This class combines physical component properties with RF-specific metrics,
      making it suitable for modeling and analyzing RF devices in a physical context.
    """

    network: FrequencyNetworkModel | None = None
    metrics: FrequencyMetricsCollection = FrequencyMetricsCollection()
