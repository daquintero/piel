from piel.types.units import *
import re


def prefix2int(s: str) -> int:
    """
    Converts a string with a number and optional suffix into an integer.

    Supported suffixes:
        'k' or 'K' - Thousand
        'm' or 'M' - Million
        'b' or 'B' - Billion
        't' or 'T' - Trillion

    Examples:
        '17.03k' -> 17030
        '17K'    -> 17000
        '2.5M'   -> 2500000
        '500'    -> 500
        '-3.2B'  -> -3200000000

    Args:
        s (str): The string to convert.

    Returns:
        int: The integer representation of the input string.

    Raises:
        ValueError: If the string format is invalid or contains unsupported suffixes.
    """
    if isinstance(s, str):
        pass
    elif isinstance(s, int):
        return s
    elif isinstance(s, float):
        return int(s)
    else:
        raise ValueError(
            f"The prefix format {s} is invalid or contains unsupported suffixes."
        )

    # Define suffix multipliers
    suffix_multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }

    # Clean the input string
    s_clean = s.strip().replace(",", "").lower()

    # Regular expression to match the number and optional suffix
    pattern = r"^([-+]?\d*\.?\d+)([kmbt]?)$"
    match = re.fullmatch(pattern, s_clean)

    if not match:
        raise ValueError(
            f"Invalid format: '{s}'. Expected formats like '17.03k', '2M', '500', etc."
        )

    number_str, suffix = match.groups()
    number = float(number_str)

    # Get the multiplier based on the suffix
    multiplier = suffix_multipliers.get(suffix, 1)

    # Calculate the integer value
    result = int(number * multiplier)

    return result


def match_unit_abbreviation(unit_str: str) -> Unit:
    """
    Matches a unit string to a predefined Unit instance.

    Parameters:
        unit_str (str): The unit abbreviation extracted from a column name (e.g., "s", "v", "dB").

    Returns:
        Unit: The corresponding Unit instance.

    Raises:
        ValueError: If the unit string does not match any predefined units.
    """
    # Mapping of unit abbreviations to Unit instances
    unit_mapping: dict[str, Unit] = {
        "ratio": ratio,
        "s": s,
        "us": us,
        "ns": ns,
        "ps": ps,
        "mw": mW,
        "w": W,
        "hz": Hz,
        "GHz": GHz,
        "db": dB,
        "v": V,
        "nm": nm,
        "mm2": mm2,
    }

    unit_str_lower = unit_str.lower()
    unit = unit_mapping.get(unit_str_lower)

    if unit is not None:
        return unit
    else:
        raise ValueError(f"Unknown unit abbreviation: '{unit_str}'")
