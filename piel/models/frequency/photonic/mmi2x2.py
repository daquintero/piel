"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""
import sax

__all__ = ["mmi2x2_50_50"]


def mmi2x2_50_50():
    S = {
        ("o1", "o3"): (1 - 0.5) ** 0.5,
        ("o1", "o4"): 1j * 0.5**0.5,
        ("o2", "o3"): 1j * 0.5**0.5,
        ("o2", "o4"): (1 - 0.5) ** 0.5,
    }
    return sax.reciprocal(S)


def mmi2x2(splitting_ratio=0.5):
    S = {
        ("o1", "o3"): (1 - splitting_ratio) ** 0.5,
        ("o1", "o4"): 1j * splitting_ratio**0.5,
        ("o2", "o3"): 1j * splitting_ratio**0.5,
        ("o2", "o4"): (1 - splitting_ratio) ** 0.5,
    }
    return sax.reciprocal(S)
