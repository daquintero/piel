import jax.numpy as jnp
from .types import ArrayTypes

__all__ = [
    "round_complex_array",
]


def round_complex_array(
    array: ArrayTypes,
    to_absolute: bool = False,
):
    """
    Rounds the elements of a complex JAX numpy array to the nearest integer.

    Parameters:
    - array: A complex JAX numpy array.
    - absolute: A boolean that determines whether the complex numbers are rounded to the nearest integers in their absolute value.

    Returns:
    - A JAX numpy array with the complex elements rounded to the nearest integers.
    """
    real_part = jnp.around(array.real)  # Round the real parts to the nearest integer
    imaginary_part = jnp.around(
        array.imag
    )  # Round the imaginary parts to the nearest integer
    value = real_part + 1j * imaginary_part  # Recombine the real and imaginary parts

    if to_absolute:
        value = jnp.abs(value)  # Take the absolute value of the complex number

    return value
