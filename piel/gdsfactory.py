"""
There are a number of ways to generate gdsfactory integration.

It is worth noting that GDSFactory has already the following PDKs installed:
* SKY130nm https://gdsfactory.github.io/skywater130/
* GF180nm https://gdsfactory.github.io/gf180/

"""

import gdsfactory as gf

def create_gdsfactory_component(
    design_directory:str,
) -> gf.Component:
    """
    This function cretes a gdsfactory component that can be included in the network codesign of the device, or that can be used for interconnection codesign.
    """
    pass


