"""
The purpose of this document is to create the generic measurement of amplifers that can be extended of integrated with more complex modelling functionality.
"""

from piel.types import RFTwoPortAmplifier


def two_port_amplifier(**kwargs) -> RFTwoPortAmplifier:
    return RFTwoPortAmplifier(**kwargs)
