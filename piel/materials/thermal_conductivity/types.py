from typing import Literal, Optional

__all__ = [
    "MaterialReferenceType",
    "MaterialReferencesTypes",
    "SpecificationType",
]

SpecificationType = Literal
MaterialReferenceType = tuple[str, Optional[str]]
MaterialReferencesTypes = list[MaterialReferenceType]
