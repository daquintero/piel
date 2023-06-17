# Python-Filesystem Useful Commands

You might want to interact a lot with the filesystem when running OSIC EDA tools, so here is a set of useful commands that might make the experience easier.

Important Libraries
```python
import os # Operating System Utilities
import pathlib # Path and directory utilities
import shutil # Shell utilities
```

## Useful Commands Table

| Description                                                | Command                                            |
|------------------------------------------------------------|----------------------------------------------------|
| Check if "examplepath" directory exists                    | `pathlib.Path.exists("<examplepath>")`             |
| Get absolute PATH of current running file                  | `pathlib.Path(__file__).resolve()`                 |
| Get absolute PATH of the directory of current running file | `pathlib.Path(__file__).parent.resolve()`          |
| Get current working directory PATH                         | `pathlib.Path(".")`                                |
| Get environment variable                                   | `os.environ["<variablename>"]`                     |
| Get POSIX representation of PATH                           | `pathlib.Path("<examplepath>").as_posix()`         |
| Get relative PATH of current running file                  | `pathlib.Path(__file__)`                           |
| Get string representation of PATH                          | `str(pathlib.Path("<examplepath>"))`               |
| Get subpath from existing PATH                             | `pathlib.Path("<examplepath>") / "<subdirectory>"` |
| Set environment variable                                   | `os.environ["<variablename>"]` = "<newvalue>"      |

