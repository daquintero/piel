Python-Filesystem Useful Commands
=================================

You might want to interact a lot with the filesystem when running OSIC
EDA tools, so here is a set of useful commands that might make the
experience easier.

Important Libraries

.. code:: python

   import os  # Operating System Utilities
   import pathlib  # Path and directory utilities
   import stat  # File permissions and status
   import shutil  # Shell utilities
   import subprocess  # Shell commands control

Useful Commands Table
---------------------

.. list-table:: Python File and System Operations
   :header-rows: 1

   * - Description
     - Command
   * - Copy a file from a source to a destination filepaths
     - ``shutil.copyfile(<sourcefilepath>, <destinationfilepath>)``
   * - Change a file permission to executable
     - ``file.chmod(file.stat().st_mode | stat.S_IEXEC)``
   * - Check if “examplepath” directory exists
     - ``pathlib.Path.exists("<examplepath>")``
   * - Get absolute PATH of current running file
     - ``pathlib.Path(__file__).resolve()``
   * - Get absolute PATH of the directory of current running file
     - ``pathlib.Path(__file__).parent.resolve()``
   * - Get current working directory PATH
     - ``pathlib.Path(".")``
   * - Get environment variable
     - ``os.environ["<variablename>"]``
   * - Get POSIX representation of PATH
     - ``pathlib.Path("<examplepath>").as_posix()``
   * - Get relative PATH of current running file
     - ``pathlib.Path(__file__)``
   * - Get string representation of PATH
     - ``str(pathlib.Path("<examplepath>"))``
   * - Get subpath from existing PATH
     - ``pathlib.Path("<examplepath>") / "<subdirectory>"``
   * - List all files and directories in PATH
     - ``list(pathlib.Path("<examplepath>").iterdir())``
   * - Run shell command
     - ``subprocess.call(<examplescriptpath>)``
   * - Set environment variable
     - ``os.environ["<variablename>"] = "<newvalue>"``
