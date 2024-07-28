import matplotlib.pyplot as plt
from typing import Optional
from ..types import MultiDataTimeSignal


def plot_time_signals(multi_data_time_signal: MultiDataTimeSignal):
    """
    TODO signals
    """
    for data_time_signal_i in multi_data_time_signal:
        plt.plot(data_time_signal_i.time_s, data_time_signal_i.voltage_V)


