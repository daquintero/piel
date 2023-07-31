import hdl21 as h
import hdl21.sim as hs
import numpy as np
import pandas as pd
from typing import Literal, Optional
import vlsirtools.spice as vsp
from ...config import piel_path_types
from ...file_system import return_path

__all__ = [
    "configure_ngspice_simulation",
    "configure_operating_point_simulation",
    "configure_transient_simulation",
    "run_simulation",
]


def configure_ngspice_simulation(
    run_directory: piel_path_types = ".",
):
    """
    This function configures the NGSPICE simulation for the circuit and returns a simulation class.

    Args:
        run_directory (piel_path_types): Directory where the simulation will be run

    Returns:
        simulation_options: Configured NGSPICE simulation options
    """
    run_directory = return_path(run_directory)
    simulation_options = vsp.SimOptions(
        simulator=vsp.SupportedSimulators.NGSPICE,
        fmt=vsp.ResultFormat.SIM_DATA,
        rundir=run_directory,
    )
    return simulation_options


def configure_operating_point_simulation(
    testbench: h.Module,
    **kwargs,
):
    """
    This function configures the DC operating point simulation for the circuit and returns a simulation class.

    Args:
        testbench (Module): HDL21 testbench
        **kwargs: Additional arguments to be passed to the operating point simulation such as name.

    Returns:
        Simulation: HDL21 simulation class
    """

    @hs.sim
    class Simulation:
        tb = testbench
        operating_point_tb = hs.Op(**kwargs)

    return Simulation


def configure_transient_simulation(
    testbench: h.Module,
    stop_time_s: float,
    step_time_s: float,
    **kwargs,
):
    """
    This function configures the transient simulation for the circuit and returns a simulation class.

    Args:
        testbench (Module): HDL21 testbench
        stop_time_s (float): Stop time of the simulation in seconds
        step_time_s (float): Step time of the simulation in seconds
        **kwargs: Additional arguments to be passed to the transient simulation

    Returns:
        Simulation: HDL21 simulation class
    """

    @hs.sim
    class Simulation:
        tb = testbench
        transient_tb = hs.Tran(
            tstop=stop_time_s * h.prefix.UNIT,
            tstep=step_time_s * h.prefix.UNIT,
            **kwargs,
        )

    return Simulation


def save_results_to_csv(
    results: hs.SimResult, file_name: str, save_directory: piel_path_types = "."
):
    """
    This function converts the simulation results to a pandas dataframe and saves it to a csv file.

    Args:
        directory (piel_path_types): Directory where the simulation will be run
    """

    save_directory = return_path(save_directory)
    # TODO check that there are more than one analysis
    analysis_results = results.an[0].data
    if type(next(iter(analysis_results.values()))) not in (
        list,
        dict,
        tuple,
        np.ndarray,
    ):
        # Check that dict values are scalars
        analysis_results = pd.DataFrame(analysis_results, index=[0])
    else:
        analysis_results = pd.DataFrame(analysis_results)

    analysis_results.to_csv(save_directory / (file_name + ".csv"))


def run_simulation(
    simulation: h.sim.Sim,
    simulator_name: Literal["ngspice"] = "ngspice",
    simulation_options: Optional[vsp.SimOptions] = None,
    to_csv: bool = True,
):
    """
    This function runs the transient simulation for the circuit and returns the results.

    Args:
        simulation (h.sim.Sim): HDL21 simulation class
        simulator_name (Literal["ngspice"]): Name of the simulator
        simulation_options (Optional[vsp.SimOptions]): Simulation options
        to_csv (bool): Whether to save the results to a csv file

    Returns:
        results: Simulation results
    """
    if simulator_name == "ngspice":
        if simulation_options is None:
            if not vsp.ngspice.available():
                print(
                    "NGSPICE is not available. Please install it. Check by running `ngspice` in the terminal."
                )
                return
            simulation_options = configure_ngspice_simulation()
        results = simulation.run(simulation_options)
    else:
        print("Simulator not supported.")
        return

    if to_csv:
        save_results_to_csv(results, simulation.tb.name)

    return results
