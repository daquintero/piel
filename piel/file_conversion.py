import pandas as pd
from pyDigitalWaveTools.vcd.parser import VcdParser
from .file_system import return_path
from .config import piel_path_types

__all__ = [
    "read_csv_to_pandas",
    "read_vcd_to_json",
]


def read_csv_to_pandas(file_path: piel_path_types):
    """
    This function returns a Pandas dataframe that contains all the simulation data outputted from the simulation run.
    """
    file_path = return_path(file_path)
    simulation_data = pd.read_csv(file_path)
    return simulation_data


def read_vcd_to_json(file_path: piel_path_types):
    file_path = return_path(file_path)
    with open(str(file_path.resolve())) as vcd_file:
        vcd = VcdParser()
        vcd.parse(vcd_file)
        json_data = vcd.scope.toJson()
    return json_data
