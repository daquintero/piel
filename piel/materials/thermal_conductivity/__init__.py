from .stainless_steel import stainless_steel
from .aluminum import aluminum
from .copper import copper
from .teflon import teflon
from .types import *

from .stainless_steel import material_references as stainless_steel_material_references
from .aluminum import material_references as aluminum_material_references
from .copper import material_references as copper_material_references
from .teflon import material_references as teflon_material_references

material_references = (
    stainless_steel_material_references
    + aluminum_material_references
    + copper_material_references
    + teflon_material_references
)
