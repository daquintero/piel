# Python-Filesystem Useful Commands

You might want to interact a lot with the filesystem when running OSIC EDA tools, so here is a set of useful commands that might make the experience easier.

Important Libraries
```python
import os # Operating System Utilities
import pathlib # Path and directory utilities
import shutil # Shell utilities
```

## Useful Commands Table

| Description                                                | Command                                       |
|------------------------------------------------------------|-----------------------------------------------|
| Check if "examplepath" directory exists                    | `pathlib.Path.exists(<examplepath>)`          |
| Get absolute path of current running file                  | `pathlib.Path(__file__).resolve()`            |
| Get absolute path of the directory of current running file | `pathlib.Path(__file__).parent.resolve()`     |
| Get environment variable                                   | `os.environ["<variablename>"]`                |
| Get relative path of current running file                  | `pathlib.Path(__file__)`                      |
| Set environment variable                                   | `os.environ["<variablename>"]` = "<newvalue>" |

