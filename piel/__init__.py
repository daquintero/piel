"""Top-level package for piel."""
import os
import pathlib

from .cocotb import *
from .defaults import *
from .file_system import *
from .integration import *
from .openlane import *
from .parametric import *

os.environ["PIEL_PACKAGE_DIRECTORY"] = str(
    pathlib.Path(__file__).parent.parent.resolve()
)

__author__ = """Dario Quintero"""
__email__ = "darioaquintero@gmail.com"
__version__ = "0.0.31"
