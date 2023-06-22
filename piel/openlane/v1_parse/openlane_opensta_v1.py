"""
These functions do not work currently.
"""

import pathlib
import pandas as pd


def read_file_meta_data(file_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Read the file and extract the metadata

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
    """
    file = open(file_path, "r")
    file_lines_raw = file.readlines()
    file_lines_data = pd.DataFrame({"lines": file_lines_raw})
    maximum_frame_amount, frame_meta_data = configure_frame_id(file_lines_data)
    configure_timing_data_rows(file_lines_data, maximum_frame_amount)
    return file_lines_data, maximum_frame_amount, frame_meta_data


def configure_frame_id(file_lines_data):
    file_lines_data["delimiters_line"] = file_lines_data.lines.str.contains(
        "=========="
    )
    file_lines_data["timing_data_line"] = file_lines_data.lines.str.contains(
        "---------"
    )
    frame_data = []
    frame_id_counter = 0
    parity_counter = 1
    row = 0
    for delimiter in file_lines_data["delimiters_line"]:
        if delimiter:
            if parity_counter % 2:
                frame_id_counter += 1
                parity_counter = 0
            else:
                parity_counter += 1
        frame_data.append(frame_id_counter - 1)
        row += 1
    file_lines_data["frame_id"] = frame_data
    maximum_frame_amount = frame_id_counter
    return maximum_frame_amount, file_lines_data


def configure_timing_data_rows(file_lines_data, maximum_frame_amount):
    frame_meta_data = {}
    for frame in range(maximum_frame_amount):
        timing_rows_index = file_lines_data.index[
            file_lines_data["timing_data_line"] & (file_lines_data["frame_id"] == frame)
        ].tolist()
        if len(timing_rows_index) >= 1:
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


def extract_frame_meta_data(file_lines_data):
    file_lines_data["start_point_line"] = file_lines_data.lines.str.contains(
        "Startpoint"
    )
    file_lines_data["end_point_line"] = file_lines_data.lines.str.contains("Endpoint")
    file_lines_data["path_group_line"] = file_lines_data.lines.str.contains(
        "Path Group"
    )
    file_lines_data["path_type_line"] = file_lines_data.lines.str.contains("Path Type")

    start_point_name = file_lines_data.lines[
        file_lines_data.start_point_line
    ].str.extract(r"((?<=Startpoint:\s).*?(?=\s\())")
    end_point_name = file_lines_data.lines[file_lines_data.end_point_line].str.extract(
        r"((?<=Endpoint:\s).*?(?=\s\())"
    )
    path_group_name = file_lines_data.lines[
        file_lines_data.path_group_line
    ].str.extract(r"((?<=Path Group:\s).*)")
    path_type_name = file_lines_data.lines[file_lines_data.path_type_line].str.extract(
        r"((?<=Path Type:\s).*)"
    )
    return start_point_name, end_point_name, path_group_name, path_type_name


def extract_timing_data(file_address, frame_meta_data, frame_id=0):
    timing_data = pd.read_fwf(
        file_address,
        colspecs=[(0, 6), (6, 14), (14, 22), (22, 30), (30, 38), (38, 40), (40, 100)],
        skiprows=frame_meta_data[frame_id]["start_rows_skip"],
        skipfooter=frame_meta_data[frame_id]["end_rows_skip"],
        names=["Fanout", "Cap", "Slew", "Delay", "Time", "Direction", "Description"],
    )
    timing_data["net_type"] = timing_data["Description"].str.extract(r"\(([^()]+)\)")
    timing_data["net_name"] = timing_data["Description"].str.extract(r"(.*?)\s?\(.*?\)")
    return timing_data


def calculate_propagation_delay(net_name_in, net_name_out, timing_data):
    if (
        len(
            timing_data[
                (timing_data.net_name == net_name_in) & (timing_data.net_type == "in")
            ]
        )
        > 0
    ) and (
        len(
            timing_data[
                (timing_data.net_name == net_name_out) & (timing_data.net_type == "out")
            ]
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


def run_parser(file_address):
    file_lines_data, maximum_frame_amount, frame_meta_data = read_file_meta_data(
        file_address
    )
    (
        start_point_name,
        end_point_name,
        path_group_name,
        path_type_name,
    ) = extract_frame_meta_data(file_lines_data)
    frame_timing_data = {}
    propagation_delay = {}
    for frame_id in range(maximum_frame_amount):
        frame_timing_data[frame_id] = extract_timing_data(
            file_address, frame_meta_data, frame_id
        )
        if len(start_point_name.values) > frame_id:
            frame_net_in = start_point_name.values[frame_id][0]
            frame_net_out = end_point_name.values[frame_id][0]
            propagation_delay[frame_id] = calculate_propagation_delay(
                frame_net_in, frame_net_out, frame_timing_data[frame_id]
            )
    return frame_timing_data, propagation_delay
