import pandas as pd

import piel
from ...types import ArrayTypes
from piel.types.materials import MaterialReferencesTypes, MaterialReferenceType
from piel.types.physical import TemperatureRangeTypes

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
    **kwargs,
) -> ArrayTypes:
    specification = material_reference[1]
    copper_thermal_conductivity_file = (
        piel.return_path(__file__).parent
        / "data"
        / "ofhc_copper_thermal_conductivity.csv"
    )
    if copper_thermal_conductivity_file.exists():
        pass
    else:
        print("Dataset is not found: ")
        print(copper_thermal_conductivity_file)
    thermal_conductivity_material_dataset = pd.read_csv(
        copper_thermal_conductivity_file
    )

    # Simplified coefficient extraction
    coefficients = {}
    for coeff in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
        coefficients[coeff] = thermal_conductivity_material_dataset.loc[
            thermal_conductivity_material_dataset["coefficient"] == coeff, specification
        ].values[0]

    # Calculating thermal conductivity
    numerator = (
        coefficients["a"]
        + coefficients["c"] * temperature_range_K**0.5
        + coefficients["e"] * temperature_range_K
        + coefficients["g"] * temperature_range_K**1.5
        + coefficients["i"] * temperature_range_K**2
    )

    denominator = (
        1
        + coefficients["b"] * temperature_range_K**0.5
        + coefficients["d"] * temperature_range_K
        + coefficients["f"] * temperature_range_K**1.5
        + coefficients["h"] * temperature_range_K**2
    )

    thermal_conductivity_fit = numerator / denominator

    return thermal_conductivity_fit
