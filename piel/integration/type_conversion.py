"""
This file provides a set of utilities in converting between common data types to represent information between different toolsets.
"""


__all__ = ["convert_2d_array_to_string"]


def convert_2d_array_to_string(list_2D: list[list]):
    """
    This function is particularly useful to convert digital data when it is represented as a 2D array into a set of strings.

    Args:
        list_2D (list[list]): A 2D array of binary data.

    Returns:
        binary_string (str): A string of binary data.

    Usage:

        list_2D=[[0], [0], [0], [1]]
        convert_2d_array_to_string(list_2D)
        >>> "0001"
    """
    binary_string = "".join(str(sublist[0]) for sublist in list_2D)
    return binary_string
