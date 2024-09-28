# Top Level Types Declaration, all should be imported here.
from piel.types.analogue import AnalogueModule, AnalogModule

from piel.types.core import (
    PathTypes,
    PielBaseModel,
    NumericalTypes,
    MinimumMaximumType,
    ArrayTypes,
    QuantityType,
    TupleIntType,
    TupleFloatType,
    TupleNumericalType,
    PackageArrayType,
    ModuleType,
)

from piel.types.connectivity.core import Instance
from piel.types.connectivity.abstract import Connection, Component, Port
from piel.types.connectivity.generic import (
    ConnectionTypes,
    PortTypes,
    ComponentTypes,
    ComponentCollection,
)
from piel.types.connectivity.physical import (
    PhysicalComponent,
    PhysicalConnection,
    PhysicalPort,
)
from piel.types.connectivity.metrics import ComponentMetrics
from piel.types.connectivity.timing import (
    TimeMetrics,
    DispersiveTimeMetrics,
    TimeMetricsTypes,
    ZeroTimeMetrics,
)

from piel.types.digital import (
    DigitalLogicModule,
    AbstractBitsType,
    BitsType,
    BitsList,
    DigitalRunID,
    HDLSimulator,
    HDLTopLevelLanguage,
    LogicSignalsList,
    LogicImplementationType,
    TruthTable,
    TruthTableLogicType,
)
from piel.types.digital_electro_optic import BitPhaseMap

from piel.types.environment import Environment
from piel.types.experimental import *  # NOQA: F403

from piel.types.electrical.cables import (
    CoaxialCableGeometryType,
    CoaxialCableHeatTransferType,
    CoaxialCableMaterialSpecificationType,
    DCCableGeometryType,
    DCCableHeatTransferType,
    DCCableMaterialSpecificationType,
    DCCable,
    CoaxialCable,
)

from piel.types.electrical.pcb import PCB

from piel.types.electrical.rf_calibration import Short, Open, Load, Through

from piel.types.electrical.rf_passives import (
    PowerSplitter,
    BiasTee,
)

from piel.types.electro_optic.transition import (
    FockStatePhaseTransitionType,
    OpticalStateTransitions,
    PhaseMapType,
    PhaseTransitionTypes,
    SwitchFunctionParameter,
    SParameterCollection,
)

from piel.types.electronic.core import (
    ElectronicCircuit,
    ElectronicChip,
    ElectronicCircuitComponent,
)
from piel.types.electronic.amplifier import RFTwoPortAmplifier
from piel.types.electronic.generic import RFAmplifierCollection, RFAmplifierTypes
from piel.types.electronic.hva import PowerAmplifierMetrics, PowerAmplifier
from piel.types.electronic.lna import LNAMetrics, LowNoiseTwoPortAmplifier

from piel.types.frequency.core import (
    FrequencyTransmissionModel,
    FrequencyTransmissionCollection,
    FrequencyTransmissionState,
    FrequencyTransmissionArrayState,
    FrequencyMetricsCollection,
    RFPhysicalComponent,
)

from piel.types.frequency.sax_core import (
    Array,
    Complex,
    ComplexArrayND,
    ComplexArray1D,
    FloatArray1D,
    FloatArrayND,
    IntArray1D,
    IntArrayND,
    Int,
    PortMap,
    PortCombination,
    SDict,
    SType,
    SCoo,
    SDense,
)  # For pure-sax compatibility.

from piel.types.file_system import ProjectType
from piel.types.integration import CircuitComponent
from piel.types.materials import (
    MaterialReferenceType,
    MaterialReferencesTypes,
    MaterialSpecificationType,
)

from piel.types.metrics import ScalarMetrics, ScalarMetricCollection

from piel.types.photonic import (
    PhotonicCircuitComponent,
    PortsTuple,
    OpticalTransmissionCircuit,
    RecursiveNetlist,
    SParameterMatrixTuple,
)

from piel.types.signal.core import ElectricalSignalDomains, QuantityTypesDC

from piel.types.signal.dc_data import (
    SignalDCCollection,
    SignalTraceDC,
    SignalDC,
)

from piel.types.signal.frequency import (
    TwoPortCalibrationNetworkCollection,
)

from piel.types.signal.time_data import (
    DataTimeSignalData,
    EdgeTransitionAnalysisTypes,
    MultiDataTimeSignal,
    MultiDataTimeSignalAnalysisTypes,
    DataTimeSignalAnalysisTypes,
)


from piel.types.signal.time_sources import (
    ExponentialSource,
    PulseSource,
    PiecewiseLinearSource,
    SineSource,
    SignalTimeSources,
)

from piel.types.reference import Reference

# Always last
from piel.types.type_conversion import (
    a2d,
    absolute_to_threshold,
    convert_array_type,
    convert_tuple_to_string,
    convert_2d_array_to_string,
    convert_to_bits,
    convert_dataframe_to_bits,
)

from .units import (
    Unit,
    BaseSIUnitNameList,
    A,
    dB,
    GHz,
    Hz,
    nm,
    ns,
    mm2,
    mW,
    ohm,
    ps,
    ratio,
    s,
    us,
    W,
    V,
)
