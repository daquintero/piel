"""Top-level package for piel."""
import os
import pathlib

# Libraries
from piel import components  # NOQA: F401
from piel import models  # NOQA: F401

# Functions
from .cocotb import *
from .config import *
from .defaults import *
from .file_system import *
from .gdsfactory import *
from .integration import *
from .openlane import *
from .parametric import *
from .sax import *

os.environ["PIEL_PACKAGE_DIRECTORY"] = str(
    pathlib.Path(__file__).parent.parent.resolve()
)

__author__ = """Dario Quintero"""
__email__ = "darioaquintero@gmail.com"
__version__ = "0.0.35"
