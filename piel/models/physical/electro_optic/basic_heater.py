__all__ = ["linear_phase_mapping_relationship"]


def linear_phase_mapping_relationship(
    phase_power_slope: float,
    zero_power_phase: float,
):
    """
    This function returns a function that maps the power applied to a particular heater resistor linearly. For
    example, we might start with a minimum phase mapping of (0,0) where the units are in (Watts, Phase). If we have a ridiculous arbitrary phase_power_slope of 1rad/1W, then the points in our linear mapping would be (0,0), (1,1), (2,2), (3,3), etc. This is implemented as a lambda function that takes in a power and returns a phase. The units of the power and phase are determined by the phase_power_slope and zero_power_phase. The zero_power_phase is the phase at zero power. The phase_power_slope is the slope of the linear mapping. The units of the phase_power_slope are radians/Watt. The units of the zero_power_phase are radians. The units of the power are Watts. The units of the phase are radians.

    Args:
        phase_power_slope (float): The slope of the linear mapping. The units of the phase_power_slope are radians/Watt.
        zero_power_phase (float): The phase at zero power. The units of the zero_power_phase are radians.

    Returns:
        linear_phase_mapping (function): A function that maps the power applied to a particular heater resistor linearly. The units of the power and phase are determined by the phase_power_slope and zero_power_phase. The zero_power_phase is the phase at zero power. The phase_power_slope is the slope of the linear mapping. The units of the phase_power_slope are radians/Watt. The units of the zero_power_phase are radians. The units of the power are Watts. The units of the phase are radians.
    """

    def linear_phase_mapping(power_w: float) -> float:
        """
        We create a linear interpolation based on the phase_power_slope. This function returns phase in radians.
        """
        return phase_power_slope * power_w + zero_power_phase

    return linear_phase_mapping
