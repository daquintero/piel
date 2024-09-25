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
