import pytest

# Import your custom Pydantic models and functions
from piel.types import (
    c,
)
from piel.models.transient.photonic import (  # Replace 'your_module' with the actual module name where the functions are defined
    v_g_from_n_g,
    n_g_from_v_g,
)

# ----------------------------
# Tests for v_g_from_n_g(n_g)
# ----------------------------


def test_v_g_from_n_g_typical():
    """
    Test v_g_from_n_g with typical group index values.
    """
    # Group index n_g = 1 (v_g should be c)
    n_g = 1.0
    expected_v_g = c.value / n_g
    assert v_g_from_n_g(n_g) == expected_v_g

    # Group index n_g = 2 (v_g should be c / 2)
    n_g = 2.0
    expected_v_g = c.value / n_g
    assert v_g_from_n_g(n_g) == expected_v_g

    # Group index n_g = 0.5 (v_g should be c / 0.5 = 2c)
    n_g = 0.5
    expected_v_g = c.value / n_g
    assert v_g_from_n_g(n_g) == expected_v_g


def test_v_g_from_n_g_zero():
    """
    Test v_g_from_n_g with n_g = 0, which should raise a ZeroDivisionError.
    """
    n_g = 0.0
    with pytest.raises(ZeroDivisionError):
        v_g_from_n_g(n_g)


def test_v_g_from_n_g_negative():
    """
    Test v_g_from_n_g with negative n_g, which may not be physically meaningful.
    """
    n_g = -1.0
    expected_v_g = c.value / n_g
    assert v_g_from_n_g(n_g) == expected_v_g


# ----------------------------
# Tests for n_g_from_v_g(v_g)
# ----------------------------


def test_n_g_from_v_g_typical():
    """
    Test n_g_from_v_g with typical group velocity values.
    """
    # v_g = c (n_g should be 1)
    v_g = c.value
    expected_n_g = c.value / v_g
    assert n_g_from_v_g(v_g) == expected_n_g

    # v_g = c / 2 (n_g should be 2)
    v_g = c.value / 2
    expected_n_g = c.value / v_g
    assert n_g_from_v_g(v_g) == expected_n_g

    # v_g = 2c (n_g should be 0.5)
    v_g = 2 * c.value
    expected_n_g = c.value / v_g
    assert n_g_from_v_g(v_g) == expected_n_g


def test_n_g_from_v_g_zero():
    """
    Test n_g_from_v_g with v_g = 0, which should raise a ZeroDivisionError.
    """
    v_g = 0.0
    with pytest.raises(ZeroDivisionError):
        n_g_from_v_g(v_g)


def test_n_g_from_v_g_negative():
    """
    Test n_g_from_v_g with negative v_g, which may not be physically meaningful.
    """
    v_g = -c.value
    expected_n_g = c.value / v_g
    assert n_g_from_v_g(v_g) == expected_n_g
