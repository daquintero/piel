import jax.numpy as jnp
import pandas as pd
from typing import Literal

import piel
from ...types import ArrayTypes
from .types import MaterialReferencesTypes, MaterialReferenceType
from ...models.physical.types import TemperatureRangeTypes

# CURRENT TODO: finish migrating this, add material sub names, add proper export, put into __init__

__all__ = ["copper", "material_references"]

supported_specifications = ["rrr50", "rrr100", "rrr150", "rrr300", "rrr500"]
material_references: MaterialReferencesTypes = [
    ("copper", specification_i) for specification_i in supported_specifications
]


def copper(
    temperature_range_K: TemperatureRangeTypes,
    material_reference: MaterialReferenceType,
    *args,
    **kwargs
) -> ArrayTypes:
    specification = material_reference[1]
    copper_thermal_conductivity_file = piel.return_path(
        __file__).parent / "data" / "ofhc_copper_thermal_conductivity.csv"
    assert copper_thermal_conductivity_file.exists()
    thermal_conductivity_material_dataset = pd.read_csv(
        copper_thermal_conductivity_file
    )

    # Simplified coefficient extraction
    coefficients = {}
    for coeff in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
        coefficients[coeff] = thermal_conductivity_material_dataset.loc[
            thermal_conductivity_material_dataset["coefficient"] == coeff, specification].values[0]

    # Calculating thermal conductivity
    numerator = (coefficients['a'] +
                 coefficients['c'] * temperature_range_K ** 0.5 +
                 coefficients['e'] * temperature_range_K +
                 coefficients['g'] * temperature_range_K ** 1.5 +
                 coefficients['i'] * temperature_range_K ** 2)

    denominator = (1 +
                   coefficients['b'] * temperature_range_K ** 0.5 +
                   coefficients['d'] * temperature_range_K +
                   coefficients['f'] * temperature_range_K ** 1.5 +
                   coefficients['h'] * temperature_range_K ** 2)

    thermal_conductivity_fit = numerator / denominator

    return thermal_conductivity_fit

# if self.material_name == "copper":
#     thermal_conductivity_material_dataset = pd.read_csv(
#         file_path + "/../materials/data/raw/thermal_conductivity/ofhc_copper_thermal_conductivity.csv")
#     a = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "a"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "a"][
#                   self.material_sub_name].values)
#     b = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "b"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "b"][
#                   self.material_sub_name].values)
#     c = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "c"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "c"][
#                   self.material_sub_name].values)
#     d = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "d"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "d"][
#                   self.material_sub_name].values)
#     e = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "e"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "e"][
#                   self.material_sub_name].values)
#     f = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "f"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "f"][
#                   self.material_sub_name].values)
#     g = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "g"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "g"][
#                   self.material_sub_name].values)
#     h = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "h"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "h"][
#                   self.material_sub_name].values)
#     i = nsy.C(s=thermal_conductivity_material_dataset[
#         thermal_conductivity_material_dataset["coefficient"] == "i"][self.material_sub_name].values[0],
#               n=thermal_conductivity_material_dataset[
#                   thermal_conductivity_material_dataset["coefficient"] == "i"][
#                   self.material_sub_name].values)
#     self.__thermal_conductivity_fit__ = nsy.V(n=10) ** ((a +
#                                                          c * self.temperature_range_K ** 0.5 +
#                                                          e * self.temperature_range_K +
#                                                          g * self.temperature_range_K ** 1.5 +
#                                                          i * self.temperature_range_K ** 2) /
#                                                         (1 +
#                                                          b * self.temperature_range_K ** 0.5 +
#                                                          d * self.temperature_range_K +
#                                                          f * self.temperature_range_K ** 1.5 +
#                                                          h * self.temperature_range_K ** 2)
#
# return self.__thermal_conductivity_fit__.n
