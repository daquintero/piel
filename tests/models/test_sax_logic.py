import sax
import jax.numpy as jnp


def active_lossless_straight(active_phase_rad=0.0):
    phase = active_phase_rad
    amplitude = 1.0
    transmission = amplitude * jnp.exp(1j * phase)
    S = sax.reciprocal({("o1", "o2"): transmission})
    return S


if __name__ == "__main__":
    print(active_lossless_straight(0.0))
    print(active_lossless_straight(jnp.pi))
