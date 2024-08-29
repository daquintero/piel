from __future__ import annotations

from piel.types import PielBaseModel


class Instance(PielBaseModel):
    """
    This represents the fundamental data structure of an element in connectivity
    """

    name: str = ""
    attrs: dict = dict()
