from .electronic.core import ElectronicCircuitComponent
from .photonic import PhotonicCircuitComponent

CircuitComponent = ElectronicCircuitComponent | PhotonicCircuitComponent
