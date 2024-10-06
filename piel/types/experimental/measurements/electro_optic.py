from piel.types.core import PathTypes
from piel.types.connectivity.generic import ConnectionTypes
from piel.types.experimental.measurements.core import Measurement, MeasurementCollection


class ElectroOpticDCMeasurement(Measurement):
    dc_transmission_file: PathTypes
    connection: ConnectionTypes


class ElectroOpticDCMeasurementCollection(MeasurementCollection):
    collection: list[ElectroOpticDCMeasurement]
