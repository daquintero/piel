import pandas as pd
import pathlib
from piel.file_system import return_path
from .utils import (
    contains_in_lines,
    read_file_lines,
    get_file_line_by_keyword,
    create_file_lines_dataframe,
)

__all__ = [
    "calculate_max_frame_amount",
    "calculate_propagation_delay_from_file",
    "calculate_propagation_delay_from_timing_data",
    "configure_timing_data_rows",
    "configure_frame_id",
    "filter_timing_data_by_net_name_and_type",
    "get_frame_meta_data",
    "get_frame_lines_data",
    "get_frame_timing_data",
    "get_all_timing_data_from_file",
    "read_sta_rpt_fwf_file",
]


def calculate_max_frame_amount(
    file_lines_data: pd.DataFrame,
):
    """
    Calculate the maximum frame amount based on the frame IDs in the DataFrame

    Args:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines

    Returns:
        maximum_frame_amount (int): Maximum number of frames in the file
    """
    return max(file_lines_data["frame_id"].tolist()) + 1


def calculate_propagation_delay_from_timing_data(
    net_name_in: str,
    net_name_out: str,
    timing_data: pd.DataFrame,
):
    """
    Calculate the propagation delay between two nets

    Args:
        net_name_in (str): Name of the input net
        net_name_out (str): Name of the output net
        timing_data (pd.DataFrame): Dataframe containing the timing data

    Returns:
        propagation_delay_dataframe (pd.DataFrame): Dataframe containing the propagation delay
    """
    if (
        len(filter_timing_data_by_net_name_and_type(timing_data, net_name_in, "in")) > 0
        and len(
            filter_timing_data_by_net_name_and_type(timing_data, net_name_out, "out")
        )
        > 0
    ):
        output_net_dataframe = (
            timing_data[(timing_data.net_type == "out")]
            .copy()
            .add_suffix("_out")
            .reset_index()
        )
        input_net_dataframe = (
            timing_data[(timing_data.net_type == "in")]
            .copy()
            .add_suffix("_in")
            .reset_index()
        )
        propagation_delay_dataframe = output_net_dataframe.merge(
            input_net_dataframe, left_index=True, right_index=True
        )
        propagation_delay_dataframe["propagation_delay"] = pd.to_numeric(
            output_net_dataframe.iloc[:].Time_out.values
        ) - pd.to_numeric(input_net_dataframe.iloc[:].Time_in.values)
        return propagation_delay_dataframe


def calculate_propagation_delay_from_file(
    file_path: str | pathlib.Path,
):
    """
    Calculate the propagation delay for each frame in the file

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        propagation_delay (dict): Dictionary containing the propagation delay
    """
    # TODO Check file is RPT
    file_lines_data = get_frame_lines_data(file_path)
    maximum_frame_amount = calculate_max_frame_amount(file_lines_data)
    (
        start_point_name,
        end_point_name,
        path_group_name,
        path_type_name,
    ) = get_frame_meta_data(file_lines_data)

    frame_timing_data = get_all_timing_data_from_file(file_path)
    propagation_delay = {}
    for frame_id in range(maximum_frame_amount):
        if len(start_point_name.values) > frame_id:
            frame_net_in = start_point_name.values[frame_id][0]
            frame_net_out = end_point_name.values[frame_id][0]
            propagation_delay[frame_id] = calculate_propagation_delay_from_timing_data(
                frame_net_in, frame_net_out, frame_timing_data[frame_id]
            )
    return propagation_delay


def configure_timing_data_rows(
    file_lines_data: pd.DataFrame,
):
    """
    Identify the timing data lines for each frame and creates a metadata dictionary for frames.

    Args:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines

    Returns:
        frame_meta_data (dict): Dictionary containing the frame metadata
    """
    frame_meta_data = {}
    maximum_frame_amount = calculate_max_frame_amount(file_lines_data)
    for frame in range(maximum_frame_amount):
        timing_rows_index = file_lines_data.index[
            file_lines_data["timing_data_line"] & (file_lines_data["frame_id"] == frame)
        ].tolist()
        if len(timing_rows_index) > 0:
            frame_meta_data[frame] = {
                "start_index": timing_rows_index[0],
                "end_index": timing_rows_index[-1],
                "start_rows_skip": timing_rows_index[0] + 1,
                "end_rows_skip": len(file_lines_data) - timing_rows_index[-1],
            }
        else:
            frame_meta_data[frame] = {
                "start_index": 0,
                "end_index": 0,
                "start_rows_skip": 0,
                "end_rows_skip": 0,
            }
    return frame_meta_data


def configure_frame_id(
    file_lines_data: pd.DataFrame,
):
    """
    Identify the frame delimiters and assign frame ID to each line in the file

    Args:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines

    Returns:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
    """
    file_lines_data["delimiters_line"] = file_lines_data.lines.str.contains(
        "=========="
    )
    file_lines_data["timing_data_line"] = file_lines_data.lines.str.contains(
        "---------"
    )
    frame_data = []
    frame_id_counter = 0
    parity_counter = 1
    for delimiter in file_lines_data["delimiters_line"]:
        if delimiter:
            if parity_counter % 2:
                frame_id_counter += 1
                parity_counter = 0
            else:
                parity_counter += 1
        frame_data.append(frame_id_counter - 1)
    file_lines_data["frame_id"] = frame_data
    return file_lines_data


def filter_timing_data_by_net_name_and_type(
    timing_data: pd.DataFrame, net_name: str, net_type: str
):
    """
    Filter the timing data by net name and type

    Args:
        timing_data (pd.DataFrame): DataFrame containing the timing data
        net_name (str): Net name to be filtered
        net_type (str): Net type to be filtered

    Returns:
        timing_data (pd.DataFrame): DataFrame containing the timing data
    """
    return timing_data[
        (timing_data.net_name == net_name) & (timing_data.net_type == net_type)
    ]


def get_frame_meta_data(file_lines_data):
    """
    Get the frame metadata

    Args:
        file_lines_data (pd.DataFrame): DataFrame containing the file lines

    Returns:
        start_point_name (pd.DataFrame): DataFrame containing the start point name
        end_point_name (pd.DataFrame): DataFrame containing the end point name
        path_group_name (pd.DataFrame): DataFrame containing the path group name
        path_type_name (pd.DataFrame): DataFrame containing the path type name
    """
    file_lines_data["start_point_line"] = contains_in_lines(
        file_lines_data, "Startpoint"
    )
    file_lines_data["end_point_line"] = contains_in_lines(file_lines_data, "Endpoint")
    file_lines_data["path_group_line"] = contains_in_lines(
        file_lines_data, "Path Group"
    )
    file_lines_data["path_type_line"] = contains_in_lines(file_lines_data, "Path Type")

    start_point_name = get_file_line_by_keyword(
        file_lines_data, "start_point", r"((?<=Startpoint:\s).*?(?=\s\())"
    )
    end_point_name = get_file_line_by_keyword(
        file_lines_data, "end_point", r"((?<=Endpoint:\s).*?(?=\s\())"
    )
    path_group_name = get_file_line_by_keyword(
        file_lines_data, "path_group", r"((?<=Path Group:\s).*)"
    )
    path_type_name = get_file_line_by_keyword(
        file_lines_data, "path_type", r"((?<=Path Type:\s).*)"
    )

    return start_point_name, end_point_name, path_group_name, path_type_name


def get_frame_lines_data(
    file_path: str | pathlib.Path,
):
    """
    Calculate the timing data for each frame in the file

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        file_lines_data (pd.DataFrame): DataFrame containing the file lines
    """
    file_path = return_path(file_path)
    # TODO Check file is RPT
    file_lines = read_file_lines(file_path)
    file_lines_data = create_file_lines_dataframe(file_lines)
    file_lines_data = configure_frame_id(file_lines_data)
    return file_lines_data


def get_frame_timing_data(
    file: str | pathlib.Path, frame_meta_data: dict, frame_id: int = 0
):
    """
    Extract the timing data from the file

    Args:
        file (str | pathlib.Path): Address of the file
        frame_meta_data (dict): Dictionary containing the frame metadata
        frame_id (int): Frame ID to be read

    Returns:
        timing_data (pd.DataFrame): DataFrame containing the timing data
    """
    file = return_path(file)
    timing_data = read_sta_rpt_fwf_file(file, frame_meta_data, frame_id)
    timing_data["net_type"] = timing_data["Description"].str.extract(r"\(([^()]+)\)")
    timing_data["net_name"] = timing_data["Description"].str.extract(r"(.*?)\s?\(.*?\)")
    return timing_data


def get_all_timing_data_from_file(
    file_path: str | pathlib.Path,
):
    """
    Calculate the timing data for each frame in the file

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        frame_timing_data (dict): Dictionary containing the timing data for each frame
    """
    file_lines_data = get_frame_lines_data(file_path)
    maximum_frame_amount = calculate_max_frame_amount(file_lines_data)
    frame_meta_data = configure_timing_data_rows(file_lines_data)
    frame_timing_data = {}
    for frame_id in range(maximum_frame_amount):
        frame_timing_data[frame_id] = get_frame_timing_data(
            file_path, frame_meta_data, frame_id
        )
    return frame_timing_data


def read_sta_rpt_fwf_file(
    file: str | pathlib.Path,
    frame_meta_data: dict,
    frame_id: int = 0,
):
    """
    Read the fixed width file and return a DataFrame

    Args:
        file (str | pathlib.Path): Address of the file
        frame_meta_data (dict): Dictionary containing the frame metadata
        frame_id (int): Frame ID to be read

    Returns:
        file_data (pd.DataFrame): DataFrame containing the file data
    """
    file = return_path(file)
    file_data = pd.read_fwf(
        str(file.resolve()),
        colspecs=[(0, 6), (6, 14), (14, 22), (22, 30), (30, 38), (38, 40), (40, 100)],
        skiprows=frame_meta_data[frame_id]["start_rows_skip"],
        skipfooter=frame_meta_data[frame_id]["end_rows_skip"],
        names=["Fanout", "Cap", "Slew", "Delay", "Time", "Direction", "Description"],
    )
    return file_data
