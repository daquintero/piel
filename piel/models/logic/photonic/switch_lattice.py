import numpy as np


def compose_switch_position_list(
    network: np.array,
    gap_elements: list = None,
    cross_elements: list = None,
    *args,
    **kwargs
):
    """
    This function returns a list of the switch positions in the network, the corresponding instance, and the 2D position in the network.

    Args:
        network (np.array): The network array.
        gap_elements (list, optional): The gap elements in the network. Defaults to None.
        cross_elements (list, optional): The cross elements in the network. Defaults to None.

    Returns:
        switch_position_list (list): A list of tuples of the form (switch_instance, (row, col)).
    """
    if cross_elements is None:
        cross_elements = ["-"]
    if gap_elements is None:
        gap_elements = ["0"]

    # Temporary fix for the case where the gap_elements and cross_elements are lists
    cross_elements = cross_elements[0]
    gap_elements = gap_elements[0]

    switch_position_list = [
        (value, (row, col))
        for row, row_values in enumerate(network)
        for col, value in enumerate(row_values)
        if (value != gap_elements)
        if (value != cross_elements)
    ]
    return switch_position_list
