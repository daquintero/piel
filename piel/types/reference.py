from piel.types import PielBaseModel


class Reference(PielBaseModel):
    text: str | None = None
    bibtex: str | None = None
