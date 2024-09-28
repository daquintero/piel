import pytest
from piel.units import get_unit_by_datum, prefix2int, match_unit_abbreviation
from piel.types.units import (
    Unit,
    A,
    dB,
    GHz,
    Hz,
    nm,
    ns,
    mm2,
    mW,
    ohm,
    ps,
    ratio,
    s,
    us,
    W,
    V,
)


def test_prefix2int():
    # Test valid inputs with suffixes
    assert prefix2int("17.03k") == 17030
    assert prefix2int("17K") == 17000
    assert prefix2int("2.5M") == 2500000
    assert prefix2int("-3.2B") == -3200000000
    assert prefix2int("1T") == 1_000_000_000_000

    # Test valid inputs without suffixes
    assert prefix2int("500") == 500
    assert prefix2int("-100") == -100

    # Test integer and float inputs
    assert prefix2int(1000) == 1000
    assert prefix2int(1000.5) == 1000

    # Test inputs with spaces and commas
    assert prefix2int(" 1,000 ") == 1000
    assert prefix2int("2,500k") == 2_500_000

    # Test invalid inputs
    with pytest.raises(ValueError):
        prefix2int("invalid")
    with pytest.raises(ValueError):
        prefix2int("123X")
    with pytest.raises(ValueError):
        prefix2int(None)


def test_match_unit_abbreviation():
    # Test valid unit abbreviations
    assert match_unit_abbreviation("s") == s
    assert match_unit_abbreviation("us") == us
    assert match_unit_abbreviation("ns") == ns
    assert match_unit_abbreviation("ps") == ps
    assert match_unit_abbreviation("mw") == mW
    assert match_unit_abbreviation("w") == W
    assert match_unit_abbreviation("hz") == Hz
    assert match_unit_abbreviation("db") == dB
    assert match_unit_abbreviation("v") == V
    assert match_unit_abbreviation("nm") == nm
    assert match_unit_abbreviation("mm2") == mm2
    assert match_unit_abbreviation("ratio") == ratio

    # Test case insensitivity
    assert match_unit_abbreviation("S") == s
    assert match_unit_abbreviation("Us") == us

    # Test invalid unit abbreviation
    with pytest.raises(ValueError):
        match_unit_abbreviation("invalid")
    with pytest.raises(ValueError):
        match_unit_abbreviation("")


def test_get_unit_by_datum():
    # Test valid datum inputs
    assert get_unit_by_datum("voltage") == V
    assert get_unit_by_datum("Voltage") == V
    assert get_unit_by_datum("second") == s
    assert get_unit_by_datum("watt") == W
    assert get_unit_by_datum("Hertz") == Hz
    assert get_unit_by_datum("meter") == nm  # Assuming nm is representative for 'meter'

    # Test units with unique data
    assert get_unit_by_datum("ampere") == A
    assert get_unit_by_datum("resistance") == ohm
    assert get_unit_by_datum("dB") == dB

    # Test case insensitivity
    assert get_unit_by_datum("VOLTAGE") == V

    # Test invalid datum input
    assert get_unit_by_datum("unknown") is None
    assert get_unit_by_datum("") is None
