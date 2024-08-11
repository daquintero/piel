import pytest
import jax.numpy as jnp
from piel.materials.thermal_conductivity import aluminum


@pytest.mark.parametrize(
    "temperature, expected",
    [
        (
            300,
            211.78811,
        ),  # Expected value needs to be calculated or verified beforehand
        (400, 236.965),  # Example expected values, replace with correct ones
    ],
)
def test_aluminum_correct_calculations(temperature, expected):
    material_ref = ("aluminum", "1100")
    result = aluminum(temperature, material_ref)
    assert jnp.isclose(
        result, expected, rtol=1e-4
    ), f"Expected {expected}, got {result}"


def test_aluminum_invalid_specification():
    with pytest.raises(ValueError):
        aluminum(300, ("aluminum", "9999"))


@pytest.mark.parametrize(
    "temperature",
    [
        300,  # scalar
        jnp.array([300, 350]),  # JAX array
    ],
)
def test_aluminum_input_types(temperature):
    material_ref = ("aluminum", "1100")
    result = aluminum(temperature, material_ref)
    assert isinstance(result, jnp.ndarray), "Result should be a JAX array"
