from typing import Optional
from piel.types import DataTimeSignalAnalysisTypes, Instance, Unit, ratio


class ParsedColumnInfo(Instance):
    analysis_type: DataTimeSignalAnalysisTypes = "delay"
    unit: Unit = ratio
    channels: str = ""
    index: Optional[int] = 0
