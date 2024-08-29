"""Top-level package for piel."""

import os
import pathlib

# Libraries - Do not change the order
from . import develop as develop
from . import materials as materials  # NOQA: F401
from . import models as models  # NOQA: F401
from . import types as types  # NOQA: F401
from . import visual as visual  # NOQA: F401
from . import tools as tools  # NOQA: F401
from . import integration as integration  # NOQA: F401
from . import units as units  # NOQA: F401

# Functions
from piel.file_system import *
from piel.project_structure import *
from piel.utils import *
from piel.connectivity import *

import piel.flows as flows  # NOQA: F401

os.environ["PIEL_PACKAGE_DIRECTORY"] = str(
    pathlib.Path(__file__).parent.parent.resolve()
)

create_piel_home_directory()  # NOQA: F405

__author__ = """Dario Quintero"""
__email__ = "darioaquintero@gmail.com"
__version__ = "0.0.56"
