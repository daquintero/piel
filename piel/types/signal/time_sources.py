from pydantic import Field
from ..core import PielBaseModel
from typing import List, Optional


class PulseSource(PielBaseModel):
    voltage_1_V: float = Field(..., description="Initial voltage level (V1) in volts")
    voltage_2_V: float = Field(..., description="Pulsed voltage level (V2) in volts")
    delay_time_s: float = Field(..., description="Delay time (TD) in seconds")
    rise_time_s: float = Field(..., description="Rise time (TR) in seconds")
    fall_time_s: float = Field(..., description="Fall time (TF) in seconds")
    pulse_width_s: float = Field(..., description="Pulse width (PW) in seconds")
    period_s: float = Field(..., description="Period of the pulse (PER) in seconds")


class SineSource(PielBaseModel):
    offset_voltage_V: float = Field(..., description="Offset voltage (VO) in volts")
    amplitude_V: float = Field(..., description="Amplitude (VA) in volts")
    frequency_Hz: float = Field(..., description="Frequency (FREQ) in hertz")
    delay_time_s: Optional[float] = Field(0.0, description="Delay time (TD) in seconds")
    damping_factor: Optional[float] = Field(0.0, description="Damping factor (THETA)")


class PiecewiseLinearSource(PielBaseModel):
    time_voltage_pairs: List[tuple] = Field(
        ..., description="List of (time, voltage) pairs"
    )


class ExponentialSource(PielBaseModel):
    voltage_1_V: float = Field(..., description="Initial voltage level (V1) in volts")
    voltage_2_V: float = Field(..., description="Pulsed voltage level (V2) in volts")
    rise_delay_time_s: float = Field(
        ..., description="Rise delay time (TD1) in seconds"
    )
    rise_time_constant_s: float = Field(
        ..., description="Rise time constant (TAU1) in seconds"
    )
    fall_delay_time_s: float = Field(
        ..., description="Fall delay time (TD2) in seconds"
    )
    fall_time_constant_s: float = Field(
        ..., description="Fall time constant (TAU2) in seconds"
    )


SignalTimeSources = PulseSource | SineSource | PiecewiseLinearSource | ExponentialSource

if __name__ == "__init__":
    # Example usage
    pulse = PulseSource(
        voltage_1_V=0,
        voltage_2_V=5,
        delay_time_s=1e-3,
        rise_time_s=1e-6,
        fall_time_s=1e-6,
        pulse_width_s=1e-3,
        period_s=2e-3,
    )
    sine = SineSource(
        offset_voltage_V=0,
        amplitude_V=5,
        frequency_Hz=50,
        delay_time_s=0.01,
        damping_factor=0.1,
    )
    pwl = PiecewiseLinearSource(time_voltage_pairs=[(0, 0), (1e-3, 5), (2e-3, 0)])
    exp = ExponentialSource(
        voltage_1_V=0,
        voltage_2_V=5,
        rise_delay_time_s=1e-3,
        rise_time_constant_s=1e-6,
        fall_delay_time_s=2e-3,
        fall_time_constant_s=1e-6,
    )

    print(pulse)
    print(sine)
    print(pwl)
    print(exp)
