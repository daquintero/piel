"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""
from ....config import nso

__all__ = ["grating_coupler_simple"]


def grating_coupler_simple(R=0.0, R_in=0.0, Tmax=1.0, bandwidth=0.06e-6, wl0=1.55e-6):
    # Constants
    fwhm2sigma = 1.0 / (2 * nso.sqrt(2 * nso.log(2)))

    # Compute sigma
    sigma = fwhm2sigma * bandwidth

    # Assume the wavelength of the environment matches the center wavelength of the grating coupler
    wls = wl0

    # Compute loss
    loss = nso.sqrt(Tmax * nso.exp(-((wl0 - wls) ** 2) / (2 * sigma**2)))

    # Create scattering dictionary
    sdict = {
        ("in0", "out1"): loss,
        ("in1", "out0"): loss,
        ("in0", "out0"): R_in,
        ("in1", "out1"): R,
    }

    return sdict
