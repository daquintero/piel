from functools import partial
from piel.types import PhysicalComponent


def physical_snspd(model: str = "SNSPD", **kwargs) -> PhysicalComponent:
    """
    This function generates a representation of a Superconducting Nanowire Single-Photon Detector (SNSPD) device.

    Parameters:
    ----------
    name : str, optional
        The name of the SNSPD device. Default is an empty string.
    model : str, optional
        The model of the SNSPD device. Default is "SNSPD".
    manufacturer : str, optional
        The manufacturer of the SNSPD device. Default is an empty string.

    Returns:
    -------
    PhysicalComponent
        An instance of the PhysicalComponent class representing the SNSPD device.
    """
    return PhysicalComponent(model=model, **kwargs)


photonspot_snspd = partial(physical_snspd, model="SNSPD", manufacturer="Photonspot")
