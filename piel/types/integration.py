from .electronic import ElectronicCircuitComponent
from .photonic import PhotonicCircuitComponent

CircuitComponent = ElectronicCircuitComponent | PhotonicCircuitComponent
