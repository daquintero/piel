import pandas as pd
import pathlib
from piel.file_system import check_path_exists, return_path

__all__ = [
    "contains_in_lines",
    "create_file_lines_dataframe",
    "get_file_line_by_keyword",
    "read_file_lines",
]


def contains_in_lines(
    file_lines_data: pd.DataFrame,
    keyword: str,
):
    """
    Check if the keyword is contained in the file lines

    Args:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
        keyword (str): Keyword to search for

    Returns:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
    """
    return file_lines_data.lines.str.contains(keyword)


def create_file_lines_dataframe(file_lines_raw):
    """
    Create a DataFrame from the raw lines of a file

    Args:
        file_lines_raw (list): list containing the file lines

    Returns:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
    """
    return pd.DataFrame({"lines": file_lines_raw})


def get_file_line_by_keyword(
    file_lines_data: pd.DataFrame,
    keyword: str,
    regex: str,
):
    """
    Extract the data from the file lines using the given keyword and regex

    Args:
        file_lines_data (pd.DataFrame): Dataframe containing the file lines
        keyword (str): Keyword to search for
        regex (str): Regex to extract the data

    Returns:
        extracted_values (pd.DataFrame): Dataframe containing the extracted values
    """
    lines_with_keyword = file_lines_data.lines[file_lines_data[f"{keyword}_line"]]
    extracted_values = lines_with_keyword.str.extract(regex)
    return extracted_values


def read_file_lines(file_path: str | pathlib.Path):
    """
    Extract lines from the file

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        file_lines_raw (list): list containing the file lines
    """
    file_path = return_path(file_path)
    file = read_file(file_path)
    return file.readlines()


def read_file(file_path: str | pathlib.Path):
    """
    Read the file from the given path

    Args:
        file_path (str | pathlib.Path): Path to the file

    Returns:
        file: the opened file
    """
    file_path = return_path(file_path)
    check_path_exists(file_path)
    return open(str(file_path.resolve()), "r")
