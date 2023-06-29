from typing import Literal

__all__ = ["get_input_ports_index"]


def get_input_ports_index(
    sorting_algorithm: Literal["prefix"] = "prefix",
    prefix: str = "in",
):
    """
    This function returns the input ports of a component. However, input ports may have different sets of prefixes and suffixes. This function implements different sorting algorithms for different ports names. The default algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple order is determined according to the last numerical index order of the port numbering.
    """
    pass
