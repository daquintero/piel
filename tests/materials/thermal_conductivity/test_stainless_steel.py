import pytest
import jax.numpy as jnp
from piel.materials.thermal_conductivity import stainless_steel


@pytest.mark.parametrize(
    "temperature, specification, expected",
    [
        (300, ("stainless_steel", "304"), 15.308),  # Example expected value
        (400, ("stainless_steel", "310"), 13.542),  # Example expected value
        (300, ("stainless_steel", "316"), 15.308),  # Example expected value
    ],
)
def test_stainless_steel_valid_specifications(temperature, specification, expected):
    result = stainless_steel(temperature, specification)
    assert jnp.isclose(
        result, expected, rtol=1e-4
    ), f"Expected {expected}, got {result}"


def test_stainless_steel_invalid_specification():
    with pytest.raises(ValueError):
        stainless_steel(300, ("stainless_steel", "999"))


@pytest.mark.parametrize(
    "temperature, specification",
    [
        (jnp.array([300, 350]), ("stainless_steel", "304")),
        (jnp.array([300, 350, 400]), ("stainless_steel", "310")),
    ],
)
def test_stainless_steel_temperature_array_input(temperature, specification):
    results = stainless_steel(temperature, specification)
    assert isinstance(results, jnp.ndarray), "Expected results to be a JAX array"
    assert results.shape == temperature.shape, "Result array should match input shape"


# Add additional tests to cover edge cases or specific behaviors
