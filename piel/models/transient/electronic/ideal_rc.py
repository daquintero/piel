import jax.numpy as jnp
from typing import Optional
from .types import RCMultiStageConfigurationType

__all__ = ["calculate_multistage_rc_performance"]


def calculate_multistage_rc_performance(
    multistage_configuration: Optional[RCMultiStageConfigurationType] = None,
    switching_frequency_Hz: Optional[float] = 1e5,
):
    """
    Calculates the total energy and power consumption for charging and discharging
    in a multistage RC configuration, as well as the transition energy and power consumption.

    Parameters:
        multistage_configuration(Optional[RCMultiStageConfigurationType]): A list of dictionaries containing the voltage and capacitance for each stage.
        switching_frequency_Hz(Optional[float]): The switching frequency of the RC stages.

    Returns:
        A tuple containing:
        - Total charge and discharge energy.
        - Total charge and discharge power consumption.
        - Transition energy.
        - Transition power consumption.
    """
    if multistage_configuration is None:
        power_level_configuration = (1.8, 3.3, 10)
        multistage_configuration = list[
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[0], "capacitance": 1e-12},
            {"voltage": power_level_configuration[1], "capacitance": 3e-12},
            {"voltage": power_level_configuration[1], "capacitance": 3e-12},
            {"voltage": power_level_configuration[2], "capacitance": 10e-12},
        ]

    # Create JAX arrays from the multistage_configuration
    voltage_array = jnp.array([stage["voltage"] for stage in multistage_configuration])
    capacitance_array = jnp.array(
        [stage["capacitance"] for stage in multistage_configuration]
    )

    # Calculate total charge discharge energy
    total_charge_discharge_energy = jnp.sum(voltage_array**2 * capacitance_array)
    total_charge_discharge_power_consumption = (
        total_charge_discharge_energy * switching_frequency_Hz
    )
    transition_energy = total_charge_discharge_power_consumption / 2
    transition_power_consumption = total_charge_discharge_power_consumption / 2

    return (
        total_charge_discharge_energy,
        total_charge_discharge_power_consumption,
        transition_energy,
        transition_power_consumption,
    )
