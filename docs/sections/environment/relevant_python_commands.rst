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

+-----------------------------------+----------------------------------+
| Description                       | Command                          |
+===================================+==================================+
| Copy a file from a source to a    | ``shutil.copyfile(<sourcefi      |
| destination filepaths             | lepath>, <destinationfilepath>`` |
+-----------------------------------+----------------------------------+
| Change a file permission to       | ``file.chmod(file.stat(          |
| executable                        | ).st_mode &#124; stat.S_IEXEC)`` |
+-----------------------------------+----------------------------------+
| Check if “examplepath” directory  | ``pathli                         |
| exists                            | b.Path.exists("<examplepath>")`` |
+-----------------------------------+----------------------------------+
| Get absolute PATH of current      | ``pa                             |
| running file                      | thlib.Path(__file__).resolve()`` |
+-----------------------------------+----------------------------------+
| Get absolute PATH of the          | ``pathlib.P                      |
| directory of current running file | ath(__file__).parent.resolve()`` |
+-----------------------------------+----------------------------------+
| Get current working directory     | ``pathlib.Path(".")``            |
| PATH                              |                                  |
+-----------------------------------+----------------------------------+
| Get environment variable          | ``os.environ["<variablename>"]`` |
+-----------------------------------+----------------------------------+
| Get POSIX representation of PATH  | ``pathlib.Pa                     |
|                                   | th("<examplepath>").as_posix()`` |
+-----------------------------------+----------------------------------+
| Get relative PATH of current      | ``pathlib.Path(__file__)``       |
| running file                      |                                  |
+-----------------------------------+----------------------------------+
| Get string representation of PATH | ``str(                           |
|                                   | pathlib.Path("<examplepath>"))`` |
+-----------------------------------+----------------------------------+
| Get subpath from existing PATH    | ``pathlib.Path("<exa             |
|                                   | mplepath>") / "<subdirectory>"`` |
+-----------------------------------+----------------------------------+
| List all files and directories in | ``list(pathlib.Pa                |
| PATH                              | th("<examplepath>").iterdir())`` |
+-----------------------------------+----------------------------------+
| Run shell command                 | ``subpro                         |
|                                   | cess.call(<examplescriptpath>)`` |
+-----------------------------------+----------------------------------+
| Set environment variable          | ``os.environ["<                  |
|                                   | variablename>"] = "<newvalue>"`` |
+-----------------------------------+----------------------------------+
