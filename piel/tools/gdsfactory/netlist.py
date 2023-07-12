from typing import Literal, Optional

__all__ = ["get_input_ports_index", "get_matched_ports_tuple_index"]


def get_matched_ports_tuple_index(
    ports_index: dict,
    selected_ports_tuple: Optional[tuple] = None,
    sorting_algorithm: Literal["prefix", "selected_ports"] = "prefix",
    prefix: str = "in",
) -> (tuple, tuple):
    """
    This function returns the input ports of a component. However, input ports may have different sets of prefixes
    and suffixes. This function implements different sorting algorithms for different ports names. The default
    algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple
    order is determined according to the last numerical index order of the port numbering. Returns just a tuple of
    the index.

    .. code-block:: python

        raw_ports_index = {
            "in_o_0": 0,
            "out_o_0": 1,
            "out_o_1": 2,
            "out_o_2": 3,
            "out_o_3": 4,
            "in_o_1": 5,
            "in_o_2": 6,
            "in_o_3": 7,
        }

        get_input_ports_tuple_index(ports_index=raw_ports_index)

        # Output
        (0, 5, 6, 7)

    Args:
        ports_index (dict): The ports index dictionary.
        selected_ports_tuple (tuple, optional): The selected ports tuple. Defaults to None.
        sorting_algorithm (Literal["prefix"], optional): The sorting algorithm to use. Defaults to "prefix".
        prefix (str, optional): The prefix to use for the sorting algorithm. Defaults to "in".

    Returns:
        matches_ports_index_tuple_order(tuple): The ordered input ports index tuple.
        matched_ports_name_tuple_order(tuple): The ordered input ports name tuple.

    """
    # TODO optimise this computaiton
    ports_index_list = list(ports_index.keys())
    matches_ports_index_tuple_order = tuple()
    matched_ports_list = tuple()
    if sorting_algorithm == "prefix":
        matched_ports_list = [
            item for item in ports_index_list if item.startswith(prefix)
        ]
        # This sorts in numerical order
        matched_ports_list.sort()
        matched_ports_name_tuple_order = tuple(matched_ports_list)
        matches_ports_index_tuple_order = tuple(
            [ports_index[port] for port in matched_ports_list]
        )
    elif sorting_algorithm == "selected_ports":
        matched_ports_list = selected_ports_tuple
        matched_ports_name_tuple_order = tuple(matched_ports_list)
        matches_ports_index_tuple_order = tuple(
            [ports_index[port] for port in matched_ports_list]
        )
    else:
        raise ValueError(
            "matches_ports_index_tuple_order "
            + str(matches_ports_index_tuple_order)
            + " not defined."
        )
    return matches_ports_index_tuple_order, matched_ports_name_tuple_order


def get_input_ports_index(
    ports_index: dict,
    sorting_algorithm: Literal["prefix"] = "prefix",
    prefix: str = "in",
) -> tuple:
    """
    This function returns the input ports of a component. However, input ports may have different sets of prefixes and suffixes. This function implements different sorting algorithms for different ports names. The default algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple order is determined according to the last numerical index order of the port numbering.

    .. code-block:: python

        raw_ports_index = {
            "in_o_0": 0,
            "out_o_0": 1,
            "out_o_1": 2,
            "out_o_2": 3,
            "out_o_3": 4,
            "in_o_1": 5,
            "in_o_2": 6,
            "in_o_3": 7,
        }

        get_input_ports_index(ports_index=raw_ports_index)

        # Output
        ((0, "in_o_0"), (5, "in_o_1"), (6, "in_o_2"), (7, "in_o_3"))

    Args:
        ports_index (dict): The ports index dictionary.
        sorting_algorithm (Literal["prefix"], optional): The sorting algorithm to use. Defaults to "prefix".
        prefix (str, optional): The prefix to use for the sorting algorithm. Defaults to "in".

    Returns:
        tuple: The ordered input ports index tuple.
    """
    # TODO optimise this computaiton
    ports_index_order = tuple()
    if sorting_algorithm == "prefix":
        matched_ports_index_tuple, matched_ports_list = get_matched_ports_tuple_index(
            ports_index=ports_index, sorting_algorithm=sorting_algorithm, prefix=prefix
        )
        ports_index_order = tuple(
            zip(matched_ports_index_tuple, matched_ports_list, strict=True)
        )

    return ports_index_order
