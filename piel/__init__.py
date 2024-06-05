"""Top-level package for piel."""
import os
import pathlib

# Libraries
from piel import materials  # NOQA: F401
from piel import models  # NOQA: F401
from piel import visual  # NOQA: F401

# Functions
from .types import *
from .file_system import *
from .integration import *
from .parametric import *
from .project_structure import *
from .tools import *
from .utils import *

os.environ["PIEL_PACKAGE_DIRECTORY"] = str(
    pathlib.Path(__file__).parent.parent.resolve()
)

create_piel_home_directory()

__author__ = """Dario Quintero"""
__email__ = "darioaquintero@gmail.com"
__version__ = "0.0.56"
