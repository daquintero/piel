import numpy as np
import pandas as pd

__all__ = [
    "append_row_to_dict",
    "points_to_lines_fixed_transient",
]


def append_row_to_dict(
    data: dict,
    copy_index: int,
    set_value: dict,
):
    """
    Get all the rows of the dictionary. We want to copy and append a row at a particular index of the dictionary values.
    Operates on existing files

    Args:
        data: Dictionary of files to be appended.
        copy_index: Index of the row to be copied.
        set_value: Dictionary of values to be set at the copied index.

    Returns:
        None
    """
    keys_list = list(data.keys())
    for key in keys_list:
        # Iterates over each key
        # Gets files at key and appends into dictionary at the end
        index_length = len(data[key])
        if type(data[key]) == list:
            data[key].append(data[key][copy_index])
        elif type(data[key]) == np.ndarray:
            data[key] = np.append(data[key], data[key][copy_index])
        elif type(data[key]) == dict:
            # Assumes a key,value {index: value} form that starts from 0
            # Find length of the dictionary
            data[key][index_length] = data[key][copy_index]
        else:
            raise ValueError(
                "files[key] invalid " + str(data[key]) + " for key: " + str(key)
            )

        if key in set_value.keys():
            # If value to set in the key set of the dictionary then update copied row latest appended
            if (type(data[key]) == list) or (type(data[key]) == np.ndarray):
                data[key][-1] = set_value[key]
            elif type(data[key]) == dict:
                data[key][index_length] = set_value[key]

    return data


def points_to_lines_fixed_transient(
    data: pd.DataFrame,
    time_index_name: str,
    fixed_transient_time=1,
    return_dict: bool = False,
    ignore_rows: list = None,
):
    """
    This function converts specific steady-state point files into steady-state lines with a defined transient time in
    order to plot digital-style files.

    For example, VCD files tends to be structured in this form:

    .. code-block:: text

        #2001
        b1001 "
        b10010 #
        b1001 !
        #4001
        b1011 "
        b1011 #
        b0 !
        #6001
        b101 "

    This means that even when tokenizing the files, when visualising it in a wave plotter such as GTKWave, the signals
    get converted from token specific times to transient signals by a corresponding transient rise time. If we want
    to plot the files correspondingly in Python, it is necessary to add some form of transient signal translation.
    Note that this operates on a dataframe where the electrical time signals are clearly defined. It copies the
    corresponding steady-state files points whilst adding files points for the time-index accordingly.

    It starts by creating a copy of the initial dataframe as to not overwrite the existing files. We have an initial
    time files point that tends to start at time 0. This means we need to add a point just before the next steady
    state point transition. So what we want to do is copy the existing row and just change the time to be the
    `fixed_transient_time` before the next transition.

    Doesn't append on penultimate row.

    Args:
        dataframe: Dataframe or dictionary of files to be converted.
        time_index_name: Name of the time index column.
        fixed_transient_time: Time of the transient signal.
        return_dict: Return a dictionary instead of a dataframe.
        ignore_rows: Rows to ignore when converting to steady-state lines.

    Returns:
        Dataframe or dictionary of files with steady-state lines.
    """
    # Convert the entire row depending on the time_index_name onto an int from a str
    data[time_index_name] = data[time_index_name].astype(int)
    data = data.to_dict()
    data = data.copy()

    if ignore_rows is None:
        ignore_rows = []

    for i in range(len(data[time_index_name]) - 1):
        # Create a copy of the first row with a i+1 index.
        if i == (len(data[time_index_name]) - 2):
            # Check if on penultimate row don't append
            pass
        else:
            new_steady_time = data[time_index_name][i + 1] - fixed_transient_time
            append_row_to_dict(
                data=data, copy_index=i, set_value={time_index_name: new_steady_time}
            )

    if return_dict:
        pass
    else:
        data = pd.DataFrame(data).sort_values(by="t")

    return data
