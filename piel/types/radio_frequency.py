from piel.types.signal.frequency.transmission import FrequencyTransmissionModel
from piel.types.signal.frequency.metrics import FrequencyMetricCollection
from piel.types.connectivity.physical import PhysicalComponent


class RFPhysicalComponent(PhysicalComponent):
    """
    Represents a physical RF (Radio Frequency) component with frequency-related properties.

    This class extends the PhysicalComponent class to include RF-specific attributes.

    Attributes:
    -----------
    network : FrequencyTransmissionModel | None
        A representation of the component's frequency network, typically containing
        s-parameter data. This is currently a placeholder and may be None.

    metrics : FrequencyMetricCollection
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

    network: FrequencyTransmissionModel | None = None
    metrics: FrequencyMetricCollection = []
