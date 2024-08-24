"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""


def mmi1x2_50_50():
    import sax

    S = {
        ("o1", "o2"): 0.5**0.5,
        ("o1", "o3"): 0.5**0.5,
    }
    return sax.reciprocal(S)


def mmi1x2(splitting_ratio=0.5):
    import sax

    S = {
        ("o1", "o2"): splitting_ratio**0.5,
        ("o1", "o3"): (1 - splitting_ratio) ** 0.5,
    }
    return sax.reciprocal(S)
