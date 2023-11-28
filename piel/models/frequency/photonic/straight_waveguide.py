"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""
import sax
import jax.numpy as jnp

__all__ = [
    "active_waveguide",
    "waveguide",
    "simple_straight",
    "lossless_straight",
    "ideal_lossless_active_waveguide",
]


def waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0):
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = sax.reciprocal({("o1", "o2"): transmission})
    return sdict


def active_waveguide(
        wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0, active_phase_rad=0.0
):
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = (2 * jnp.pi * neff * length / wl) + active_phase_rad
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = sax.reciprocal({("o1", "o2"): transmission})
    return sdict


def simple_straight(length=10.0, width=0.5):
    S = {("o1", "o2"): 1.0}  # we'll improve this model later!
    return sax.reciprocal(S)


def lossless_straight():
    S = {("o1", "o2"): 1.0}  # we'll improve this model later!
    return sax.reciprocal(S)


def ideal_lossless_active_waveguide(active_phase_rad=0.0):
    phase = active_phase_rad
    amplitude = 1.0
    transmission = amplitude * jnp.exp(1j * phase)
    S = sax.reciprocal({("o1", "o2"): transmission})
    return S
