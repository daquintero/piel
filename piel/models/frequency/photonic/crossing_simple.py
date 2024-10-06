def crossing_simple(cross_transmission=0.9999):
    import sax

    """
    An ideal crosser functionality swaps the input of the diagonal connection with some crossing loss.

    .. code::

        o2 ----   ---- o3
               \ /
                X
               / \
        o1 ---    ---- o4


    Args:
        cross_transmission: TODO

    Returns:

    """
    S = {
        ("o1", "o3"): cross_transmission * 1,
        ("o1", "o4"): 0,
        ("o2", "o3"): 0,
        ("o2", "o4"): cross_transmission * 1,
    }
    return sax.reciprocal(S)
