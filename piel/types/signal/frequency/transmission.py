from __future__ import annotations
from piel.types.connectivity.abstract import Instance
from piel.types.signal.frequency.sax_core import SType
from piel.types.signal.frequency.generic import PhasorTypes
from piel.types.photonic import PortMap
from typing import Any


class PathTransmission(Instance):
    ports: PortMap
    transmission: PhasorTypes
    """
    A frequency-domain representation of transmission requires the full phasor response which can be represented
    as an array, an individual value, or a static-typed data structure - as per performance requirements.
    """


class NetworkTransmission(Instance):
    """
    This corresponds to a transmission component or array collection of the power or frequency transmission.
    For example, for the reflected power (ie S11 transmission), this contains magnitude and phase information from a source.
    Instead of responding to a given input. Note that this does not contain mode information, but could
    be extended to implement this.

    This can represent frequency to single-state conversion and a sckit-rf collective model too, based on its definition.

    This implementation is flexible because making the transmission individual is kind of essential when dealing with both electronic-photonic s-parameter state
    management, or otherwise it involves writing a mapping function. This is not the fastest approach, but certainly complete.
    Maybe someone can come up with a more complete approach that is not so resource intensive or we can abstract this into
    defined base-types (but I think this is it though if any validation is to be applied?)

    This can also be equivalent to a sckit-rf Network static data container, just that it decomposes each specific transmission to a given
    frequency or power-point. It implements translation between RF models and Photonic models which are more-port specific
    as defined by SAX. This enables more specific electronic-photonic state mapping.
    """

    input: PhasorTypes
    """
    The combined definition of the input state phasor with magnitude and phase information.
    Could be extended to a spectral input incidence. The length of the PhasorType should match the length of the
    equivalent PathTransmission for one dimension of the phasor representation.
    """

    network: list[PathTransmission] = []
    """
    Contains the entire frequency transmission response per component. Can be defined both per state and per full collection.

    TODO implement port mapping already.
    """


FrequencyTransmissionModel = NetworkTransmission | SType | Any | None
"""
Corresponds to a container that contains a s-parameter transmission model, for example.

This type alias is currently a placeholder (Any | None).
The idea is that this is a collective static data representation compatible with both a sax-translation as
with the standard sckit-rf network models.
"""
