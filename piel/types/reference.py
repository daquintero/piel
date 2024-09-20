from piel.types import PielBaseModel


class Reference(PielBaseModel):
    text: str = ""
    bibtex_id: str = ""
    bibtex: str = ""
