import numpy as np
import pandas as pd
import piel

example_random_values = pd.Series(
    np.array(
        [
            "1111",
            "10010",
            "1011",
            "1001",
            "101",
            "11",
            "10000",
            "1101",
            "10001",
            "1100",
            "11010",
        ],
        dtype=np.int64,
    )
)

four_bits_array = piel.models.logic.electro_optic.bits_array_from_bits_amount(4)

basic_ideal_phase_map = piel.models.logic.electro_optic.linear_bit_phase_map(
    bits_amount=5, final_phase_rad=np.pi, initial_phase_rad=0
)


print(example_random_values)

phase_array = piel.models.logic.electro_optic.return_phase_array_from_data_series(
    data_series=example_random_values, phase_map=basic_ideal_phase_map
)
print(phase_array)
# phase_array = []
# for code_i in example_random_values.values:
#     phase = basic_ideal_phase_map[basic_ideal_phase_map.bits == str(code_i)].phase.values[0]
#     phase_array.append(phase)
# print(phase_array)

# print(basic_ideal_phase_map[basic_ideal_phase_map.bits == example_random_values].phase.values[0])
# print(basic_ideal_phase_map[basic_ideal_phase_map.bits == "1110"].phase.values[0])
# print(basic_ideal_phase_map)
# print(type(basic_ideal_phase_map.bits.iloc[-1]))
# print(basic_ideal_phase_map.bits.astype(dtype="<U4"))
# print(basic_ideal_phase_map)
# print(type(basic_ideal_phase_map.bits.iloc[-1]))
# print(four_bits_array)
# print(four_bits_array.dtype)
