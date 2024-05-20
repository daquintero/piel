import pytest
import os
from subprocess import run, CalledProcessError


@pytest.fixture(scope="module")
def script_directory():
    return os.environ.get("PIEL_PACKAGE_DIRECTORY")


def test_script_execution(script_directory):
    scripts = [
        "00_setup.py",
        "01_run_openlane_flow.py",
        "02a_large_scale_digital_layout.py",
        "03_sax_basics.py",
        "03a_sax_cocotb_cosimulation.py",
        "03b_optical_function_verification.py",
        "04_spice_cosimulation.py",
        "04a_analogue_circuit_layout_simulation.py",
        "05_quantum_integration_basics.py",
        "06_component_codesign_basics.py",
        "07_mixed_signal_photonic_cosimulation.py",
        "08_basic_interconnection_modelling.py",
    ]

    for script in scripts:
        script_path = os.path.join(script_directory, script)
        try:
            result = run(
                ["poetry", "run", "python", script_path],
                capture_output=True,
                text=True,
                check=True,
            )
        except CalledProcessError as e:
            pytest.fail(
                f"Script {script} failed with return code {e.returncode}, output: {e.output}"
            )


if __name__ == "__main__":
    pytest.main()
