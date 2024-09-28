from piel.types.core import PielBaseModel


class SymbolicValue(PielBaseModel):
    name: str = ""
    label: str = ""
    """
    Label used in plots and more.
    """
    shorthand: str = ""
    symbol: str = ""
