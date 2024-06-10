import pytest
import numpy as np
import pandas as pd
from piel.types import (
    BitPhaseMap,
)  # Adjust the import based on your actual module structure


def test_bit_phase_map_initialization():
    # Test initialization with list
    bits = ["0101", "1100", "1010"]
    phases = [0.1, 0.5, 0.9]
    bpm = BitPhaseMap(bits=bits, phase=phases)
    assert bpm.bits == bits
    assert bpm.phase == phases

    # Test initialization with numpy arrays
    bits = np.array(["0001", "0010"])
    phases = np.array([0.2, 0.4])
    bpm = BitPhaseMap(bits=bits, phase=phases)
    assert np.array_equal(bpm.bits, bits)
    assert np.array_equal(bpm.phase, phases)


def test_bit_phase_map_dataframe_property():
    bits = ["0010", "1101"]
    phases = [0.2, 0.8]
    bpm = BitPhaseMap(bits=bits, phase=phases)
    df = bpm.dataframe

    assert isinstance(df, pd.DataFrame)
    assert "bits" in df.columns
    assert "phase" in df.columns
    assert df["bits"].tolist() == bits
    assert df["phase"].tolist() == phases


def test_bit_phase_map_invalid_input():
    bits = ["0101", "1100"]
    phases = ["invalid", "data"]  # Phases should be numerical types
    with pytest.raises(ValueError):
        BitPhaseMap(bits=bits, phase=phases)
