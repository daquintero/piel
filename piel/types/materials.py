"""
This module defines type aliases for material specifications and references used in various applications.
"""

from typing import Literal, Optional

# Type alias for material specification types, allowing for strict typing using Literal.
MaterialSpecificationType = Literal
"""
MaterialSpecificationType:
    A literal type used for defining material specifications.
    This is typically used to enforce specific string values in material specifications.
"""

# Type alias for a material reference, represented as a tuple containing a mandatory string and an optional string.
MaterialReferenceType = tuple[str, Optional[str]]
"""
MaterialReferenceType:
    A tuple used to reference materials, where:
    - The first element is a mandatory string, usually representing the material name or identifier.
    - The second element is an optional string, which might provide additional details or context about the material.
"""

# Type alias for a list of material references, each conforming to the MaterialReferenceType.
MaterialReferencesTypes = list[MaterialReferenceType]
"""
MaterialReferencesTypes:
    A list of material references, where each reference is a tuple as defined by MaterialReferenceType.
    This is used to handle collections of material references in applications.
"""
