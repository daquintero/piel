import numpy as np
import piel

test_data = {"t": np.array([3000, 4000, 5000, 6000]), "x": np.array([2, 3, 4, 5])}

print(test_data)

piel.append_row_to_dict(data=test_data, copy_index=1, set_value={"t": 10})
print(test_data)

"""
Valid output
{'t': array([3000, 4000, 5000, 6000]), 'x': array([2, 3, 4, 5])}
{'t': array([3000, 4000, 5000, 6000,   10]), 'x': array([2, 3, 4, 5, 3])}
"""

"""
Now we test that it creates a copy of the data with corresponding changed data points.
"""

test_data = {"t": np.array([3000, 4000, 5000, 6000]), "x": np.array([2, 3, 4, 5])}

out_data = piel.points_to_lines_fixed_transient(
    data=test_data,
    time_index_name="t",
    fixed_transient_time=1,
)

print(test_data)
print(out_data)

"""
Valid output
{'t': array([3000, 4000, 5000, 6000]), 'x': array([2, 3, 4, 5])}
{'t': array([3000, 4000, 5000, 6000, 3999, 4999, 5999, 3998]), 'x': array([2, 3, 4, 5, 2, 3, 4, 5])}
"""
