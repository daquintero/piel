VoltageCurrentSignalNamePair = tuple[str, str]

# from .core import Measurement
# from ....measurement import PathTypes, SignalDC
#
#
# class SourcemeterSweepMeasurement(Measurement):
#     """
#     This class is used to represent a measurement of a sweep of a sourcemeter.
#
#     The sweep file is the file that contains the sweep data.
#
#     The signal is the signal that is being sourced and measured, it is defined based on the `SignalDC` class.
#     """
#
#     sweep_file: PathTypes
#     signal: SignalDC = None
#
#
# class MultimeterSweepMeasurement(Measurement):
#     """
#     This class is used to represent a measurement of a sweep of a sourcemeter.
#
#     The sweep file is the file that contains the sweep data.
#
#     The signal is the signal that is being measured, it is defined based on the `SignalDC` class.
#     """
#
#     sweep_file: PathTypes
#     signal = SignalDC = None
