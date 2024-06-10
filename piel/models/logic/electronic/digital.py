from ....types import BitsList


def bits_array_from_bits_amount(bits_amount: int) -> BitsList:
    """
    Returns an array of bits (in bytes) of a given length.

    Args:
        bits_amount(int): Amount of bits to generate.

    Returns:
        BitsList: List of binary representations in bytes.
    """
    # Generate range of integers from 0 to 2^bits_amount - 1
    maximum_integer_represented = 2**bits_amount

    # Convert each integer to its binary representation, padded with leading zeros
    bit_array = [
        format(i, f"0{bits_amount}b")  # Convert each binary string to bytes
        for i in range(maximum_integer_represented)
    ]

    return bit_array
