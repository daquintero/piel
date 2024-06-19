"""
This module defines data models and types for working with digital circuits and logic signals.
It leverages pydantic for model validation and pandas for data manipulation.
"""

import pandas as pd
from pydantic import ConfigDict
from typing import Literal, Iterable
from .core import PielBaseModel, PathTypes

# Type aliases for different types of digital bits and HDL simulators.
AbstractBitsType = str | bytes | int
"""
AbstractBitsType: Alias for types representing digital bits.
    Can be one of:
    - str: A string representation of bits.
    - bytes: A byte representation of bits.
    - int: An integer representation of bits.
"""

BitsType = str
"""
BitsType: A type representing binary digital bits.
    It is an alias for the 'str' type.
"""

BitsList = Iterable[BitsType]
"""
BitsList: An iterable collection of AbstractBitsType elements.
    Represents a sequence of digital bits.
"""

DigitalRunID = tuple[PathTypes, str]

HDLSimulator = Literal["icarus", "verilator"]
"""
HDLSimulator: A literal type representing supported HDL simulators.
    Can be one of:
    - "icarus": Refers to Icarus Verilog simulator.
    - "verilator": Refers to Verilator simulator.
"""

HDLTopLevelLanguage = Literal["verilog", "vhdl"]
"""
HDLTopLevelLanguage: A literal type representing top-level hardware description languages.
    Can be one of:
    - "verilog": Refers to the Verilog HDL.
    - "vhdl": Refers to the VHDL HDL.
"""

LogicSignalsList = list[str]
"""
LogicSignalsList: A list of strings representing the names of logic signals.
"""

TruthTableLogicType = Literal["implementation", "full"]
LogicImplementationType = Literal["combinatorial", "sequential", "memory"]


class TruthTable(PielBaseModel):
    """
    A model representing a truth table for a digital circuit, including its input and output ports.

    Attributes:
        input_ports (LogicSignalsList): List of input signal names for the truth table.
        output_ports (LogicSignalsList): List of output signal names for the truth table.

    Properties:
        keys_list (list[str]): A combined list of input and output signal names.
        dataframe (pd.DataFrame): A pandas DataFrame representation of the truth table, excluding input and output ports.
        implementation_dictionary (dict): A dictionary including only the keys specified within input_ports and output_ports.
    """

    model_config = ConfigDict(extra="allow")

    input_ports: LogicSignalsList
    """
    input_ports (LogicSignalsList): List of input signal names for the truth table.
    """

    output_ports: LogicSignalsList
    """
    output_ports (LogicSignalsList): List of output signal names for the truth table.
    """

    @property
    def ports_list(self) -> list[str]:
        """
        Returns a combined list of input and output signal names.

        Returns:
            list[str]: The concatenated list of input and output ports.
        """
        return self.input_ports + self.output_ports

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representation of the truth table, excluding the input and output ports.

        Returns:
            pd.DataFrame: A DataFrame with the truth table data, excluding input and output port keys.
        """
        data = {
            k: v
            for k, v in self.dict().items()
            if k not in {"input_ports", "output_ports"}
        }
        return pd.DataFrame(data)

    @property
    def implementation_dictionary(self) -> dict:
        """
        Returns a dictionary including only the keys specified within input_ports and output_ports.

        Returns:
            dict: A dictionary with keys that are part of the input and output ports.
        """
        selected_ports = set(self.input_ports + self.output_ports)
        filtered_dict = {k: v for k, v in self.dict().items() if k in selected_ports}
        return filtered_dict
