"""
One of the main complexities of defining frequency domain models is the dimensionality difference between
photonic and radio-frequency s-parameter responses. Whilst, in a way, it's similar - it can also be different.

One interesting difference is possibly related to the number of higher number S matrices in photonics, which also exist in RF.
For example, in a 4-port 2x2 MMI, we are looking into the transmission and back reflection of one port with N ports, in this case 2 output ports.
This kind of means that standard 2-port devices we probe using VNAs, still can probe N modes, but it has to be higher-dimensionally post-multiple-measurements constructed. So clearly, the way we probe photonics and electronics in a way is inherently different.

When we use an array of VGAs to measure PIC ports, we can measure averaged optical power which is a spectrum to scalar conversion.
I guess it's the equivalent of an RF powermeter too? But I feel the infrastructure is more multi-port in photonics than RF broadly,
or at least at a more accessible price.

So when we define a given static-frequency transformation across a circuit, in photonics it's clearer to define this at multiple ports.
The measurement we would do with a VNA or OSA, would give us a set of network transformations at multiple frequencies.

However, if we want to create static frequency transmission models, we need to think about an individual frequency.
So ultimately, in the way we represent network transformations, there is an inherent map relationship at a given frequency.
And at that given frequency we can create a full network transmission model. In this sense, I think the fundamental
data structure has to be a dictionary, of dictionaries to represent ``port -> transmission``  responses and this is
what sax does so well in embedding this into network machine learning in this format.

A higher dimensional mapping is ``state -> port -> transmission`` where the state can correspond to a given input mode,
frequency or power yet the physical transmission mapping is equivalent. As such, this is the data structure we implement.
However, this has performance caveats and I believe computation onto this data structure should be optimised.

Possibly, this is why when we are measuring an equivalent spectrum we might want to treat the network as a set of arrays
for computational speedup. So inherently, it could be argued, for reasonable speed there might need to be transformations
in between these data types depending on whether optimizing network multi-port responses like photonics, or spectrum analysis.
It is not exactly clear to me the most fundamental way to implement this that does not have inherent computational cost.
"""

from __future__ import annotations
from piel.types.core import PielBaseModel, ArrayTypes, NumericalTypes
from piel.types.metrics import ScalarMetrics
from piel.types.connectivity.physical import PhysicalComponent
from piel.types.connectivity.abstract import Instance
from piel.types.frequency.sax_core import SType
from typing import Any
from typing_extensions import TypedDict


class FrequencyTransmissionState(TypedDict):
    """
    This corresponds to a transmission component of the power or frequency transmission.
    For example, for the reflected power (ie S11 transmission), this contains magnitude and phase information.
    Instead of responding to a given input. Note that this does not contain mode information, but could
    be extended to implement this.

    This performs frequency to single-state conversion. It does not represent a sckit-rf collective model.

    Making the transmission individual is kind of essential when dealing with both electronic-photonic s-parameter state
    manamgenet, or otherwise it involves writing a mapping function. This is not the fastest approach, but certainly complete.

    Maybe someone can come up with a more complete approach that is not so resource intensive or we can abstract this into
    defined base-types (but I think this is it though if any validation is to be applied?)
    """

    input_frequency_Hz: NumericalTypes
    p_in_dbm: NumericalTypes
    transmission: SType


class FrequencyTransmissionArrayState(Instance):
    input_frequency_Hz: ArrayTypes | None = None
    p_in_dbm: ArrayTypes | None = None
    s_11_db: ArrayTypes | None = None
    s_11_deg: ArrayTypes | None = None
    s_12_db: ArrayTypes | None = None
    s_12_deg: ArrayTypes | None = None
    s_21_db: ArrayTypes | None = None
    s_21_deg: ArrayTypes | None = None
    s_22_db: ArrayTypes | None = None
    s_22_deg: ArrayTypes | None = None


class FrequencyTransmissionCollection(PielBaseModel):
    """
    This should be equivalent to a sckit-rf Network static data container, just that it decomposes each specific transmission to a given
    frequency or power-point. It implements translation between RF models and Photonic models which are more-port specific
    as defined by SAX. This enables more specific electronic-photonic state mapping.

    TODO come up with a more resource managed version of this.
    """

    name: str = ""
    collection: list[FrequencyTransmissionState]


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


FrequencyTransmissionModel = (
    FrequencyTransmissionCollection
    | FrequencyTransmissionArrayState
    | SType
    | Any
    | None
)
"""
Corresponds to a container that contains a s-parameter transmission model, for example.

This type alias is currently a placeholder (Any | None).
The idea is that this is a collective static data representation compatible with both a sax-translation as
with the standard sckit-rf network models.
"""


class RFPhysicalComponent(PhysicalComponent):
    """
    Represents a physical RF (Radio Frequency) component with frequency-related properties.

    This class extends the PhysicalComponent class to include RF-specific attributes.

    Attributes:
    -----------
    network : FrequencyTransmissionModel | None
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

    network: FrequencyTransmissionModel | None = None
    metrics: list[FrequencyMetricsCollection] = []
