"""Top-level package for piel."""

import os
import pathlib

# Libraries - Do not move from here
import piel.develop as develop
import piel.materials as materials  # NOQA: F401
import piel.models as models  # NOQA: F401
import piel.types as types  # NOQA: F401
import piel.visual as visual  # NOQA: F401
import piel.tools as tools  # NOQA: F401
import piel.integration as integration  # NOQA: F401

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
