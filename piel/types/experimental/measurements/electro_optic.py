from piel.types.core import PathTypes
from piel.types.connectivity.abstract import PortMap
from piel.types.experimental.measurements.core import Measurement, MeasurementCollection


class ElectroOpticDCMeasurement(Measurement):
    dc_transmission_file: PathTypes
    port_map: PortMap


class ElectroOpticDCMeasurementCollection(MeasurementCollection):
    collection: list[ElectroOpticDCMeasurement]
