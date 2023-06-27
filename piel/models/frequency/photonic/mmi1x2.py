import sax


def mmi1x2_50_50():
    S = {
        ("o1", "o2"): 0.5**0.5,
        ("o1", "o3"): 0.5**0.5,
    }
    return sax.reciprocal(S)
