import subprocess
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
import threading


def run_script(py_file: Path) -> Tuple[Path, bool, str]:
    """
    Executes a single Python script from its container directory.

    Parameters:
        py_file (Path): The path to the Python script to execute.

    Returns:
        Tuple containing:
            - Path of the script.
            - Success status (True if executed successfully, False otherwise).
            - Output or error message.
    """
    try:
        # Execute the Python script from its directory
        result = subprocess.run(
            ["python", str(py_file.name)],
            cwd=py_file.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        return (py_file, True, output)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "An unknown error occurred."
        return (py_file, False, error_msg)
    except Exception as e:
        # Catch any other exceptions
        return (py_file, False, str(e))


def run_all_python_examples_two_levels_deep(
    directory: Path,
    raise_exception: bool = False,
    max_workers: int = 4,  # You can adjust the number of threads
):
    """
    Runs all Python scripts in the specified directory and its immediate subdirectories concurrently,
    excluding 'run_all_notebooks.py'. Each script is executed from its own directory.
    After execution, prints a summary of the results.

    Parameters:
        directory (Path): The path to the directory containing the Python scripts.
        raise_exception (bool): If True, raises an exception on the first script failure.
                                If False, continues running remaining scripts even if some fail.
        max_workers (int): The maximum number of threads to use for concurrent execution.
    """
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    python_files: List[Path] = []

    # Level 1: Python files in the main directory
    level1_files = sorted(directory.glob("*.py"))
    python_files.extend(level1_files)

    # Level 2: Python files in immediate subdirectories
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir():
            subdir_py_files = sorted(subdir.glob("*.py"))
            python_files.extend(subdir_py_files)

    # **Exclusion Section: Skip 'run_all_notebooks.py'**
    excluded_files = {"run_all_notebooks.py", "setup.py"}

    # Filter out the excluded files
    python_files = [f for f in python_files if f.name not in excluded_files]

    if not python_files:
        print("No Python scripts found to run.")
        return

    # Initialize counters and lists for summary
    total_scripts = len(python_files)
    succeeded_scripts = 0
    failed_scripts = 0
    failed_files: List[Path] = []
    lock = threading.Lock()  # To synchronize access to shared variables

    print(f"Starting execution of {total_scripts} Python script(s)...\n")

    # Function to handle each script's result
    def handle_result(future):
        nonlocal succeeded_scripts, failed_scripts
        py_file, success, message = future.result()
        relative_path = py_file.relative_to(directory)
        if success:
            with lock:
                succeeded_scripts += 1
            print(f"Running {relative_path}...")
            if message:
                print(message)
            print(f"{py_file.name} finished successfully.\n")
        else:
            with lock:
                failed_scripts += 1
                failed_files.append(py_file)
            print(f"Running {relative_path}...")
            print(f"Error running {relative_path}:")
            print(message)
            print()
            if raise_exception:
                print("Raising exception due to script failure.")
                executor.shutdown(wait=False)
                raise subprocess.CalledProcessError(
                    returncode=1, cmd=["python", str(py_file)], stderr=message
                )

    # Use ThreadPoolExecutor for concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scripts to the executor
        future_to_script = {
            executor.submit(run_script, py_file): py_file for py_file in python_files
        }

        try:
            # As each future completes, handle the result
            for future in concurrent.futures.as_completed(future_to_script):
                handle_result(future)
        except subprocess.CalledProcessError:
            print("Execution halted due to a script failure.")
            # Optionally, you can choose to cancel all other futures here
            # for future in future_to_script:
            #     future.cancel()
            pass

    # Print summary
    print("Execution Summary")
    print("-----------------")
    print(f"Total scripts run      : {total_scripts}")
    print(f"Successfully executed : {succeeded_scripts}")
    print(f"Failed executions     : {failed_scripts}")

    if failed_scripts > 0:
        print("\nList of failed scripts:")
        for failed_file in failed_files:
            print(f"- {failed_file.relative_to(directory)}")

    print("\nAll scripts have been processed.")


# Example usage:
if __name__ == "__main__":
    # Replace with your target directory
    target_directory = Path(__file__).parent.expanduser()
    run_all_python_examples_two_levels_deep(
        target_directory,
        raise_exception=False,  # Set to True to stop on first failure
        max_workers=4,  # Adjust based on your system's capabilities
    )
